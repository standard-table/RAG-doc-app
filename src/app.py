import streamlit as st
import google.generativeai as genai
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import sys
import inspect

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="RAG with Gemini & Phoenix Docs", layout="wide")

# --- Application Constants ---
DOCS_PATH = "dat/phx-docs/docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = 'gemini-2.5-pro-exp-03-25' # Using the specific experimental model name

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. 

Answer the following question based ONLY on the provided context from
Arize AI's Phoenix documentation files.  

If the context is not sufficient or doesn't contain the answer, clearly
state that the answer is not found in the provided information.  

When you find an answer, make sure to include the source file name and
score from the context.  

Do not use any external knowledge.

Context from Phoenix Docs:
---
{rag_context}
---

Question: {user_query}

Answer:
"""

# --- Initialize Session State ---
if 'last_sent_prompt' not in st.session_state:
    st.session_state.last_sent_prompt = "No query processed yet."
if 'editable_prompt_template' not in st.session_state:
    st.session_state.editable_prompt_template = DEFAULT_PROMPT_TEMPLATE


# --- LlamaIndex Global Settings ---
try:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
except Exception as e:
    st.error(f"Fatal Error: Could not initialize HuggingFace Embedding model ('{EMBEDDING_MODEL_NAME}'). RAG functionality will be disabled. Error: {e}")
    # Potentially st.stop() here if RAG is absolutely critical and cannot proceed
    # For now, the app will load, but retrieval will fail.

# --- Google API Key and Gemini Model Initialization ---
model = None
# Attempt to configure from environment variable first
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        # st.sidebar.info("Using Google API Key from environment variable.") # Can be noisy
    except Exception as e:
        st.sidebar.error(f"Failed to configure Gemini with environment API Key: {e}")
        model = None # Explicitly set to None on failure

# If model is not set (either env var not found or config failed), try secrets
if not model:
    try:
        SECRET_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=SECRET_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        GOOGLE_API_KEY = SECRET_API_KEY # Update GOOGLE_API_KEY if sourced from secrets
        # st.sidebar.info("Using Google API Key from Streamlit secrets.")
    except KeyError:
        # This means st.secrets["GOOGLE_API_KEY"] was not found.
        # This is expected if not using secrets.toml or if the key isn't there.
        # No specific warning here if env var also failed, as the final block will prompt for input.
        pass
    except Exception as e:
        # Catch other potential errors during secrets access or genai configuration.
        # Only show an error if it's not the typical "secrets file not found" type of message.
        error_str = str(e).lower()
        if "no secrets found" not in error_str and "could not find secrets file" not in error_str:
            st.sidebar.error(f"Error when trying to use Streamlit secrets for API Key: {e}")
        # Ensure model is None if any exception occurred here and it wasn't already configured
        if not model:
            model = None

# --- LlamaIndex Data Loading and Indexing ---
@st.cache_resource(show_spinner="Loading and indexing Phoenix docs...")
def load_phx_index(docs_path: str):
    if not Settings.embed_model:
        st.error("Embedding model not loaded. Cannot build document index.")
        return None
    try:
        reader = SimpleDirectoryReader(input_dir=docs_path, required_exts=[".md"], recursive=True)
        documents = reader.load_data()
        if not documents:
            st.error(f"No markdown documents found in '{docs_path}'. Cannot build index.")
            return None
        index = VectorStoreIndex.from_documents(documents)
        return index
    except Exception as e:
        st.error(f"Error loading or indexing documents from '{docs_path}': {e}")
        return None

# --- LlamaIndex Context Retrieval ---
def get_context_from_llamaindex(query_str: str, index_instance, top_k: int = 3) -> str:
    if index_instance is None:
        return "Error: Document index not available or failed to load."
    try:
        retriever = index_instance.as_retriever(similarity_top_k=top_k)
        retrieved_nodes: List[NodeWithScore] = retriever.retrieve(query_str)
        if not retrieved_nodes:
            return "No relevant context found in the Phoenix Docs for your query."
        
        # Format context, including score and source (filename)
        context_parts = []
        for node_with_score in retrieved_nodes:
            file_name = node_with_score.metadata.get('file_name', 'Unknown source')
            score = node_with_score.score
            content = node_with_score.get_content()
            context_parts.append(f"Source: {file_name} (Score: {score:.4f})\n{content}")
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        st.error(f"Error retrieving context from LlamaIndex: {e}")
        return "Error during context retrieval."

# --- RAG Logic ---
def generate_answer_with_rag(
    user_query: str, 
    genai_model_instance, 
    rag_context: str, 
    temperature: float,
    top_p: float,
    prompt_template: str
) -> str:
    
    final_prompt = prompt_template.format(rag_context=rag_context, user_query=user_query)
    st.session_state.last_sent_prompt = final_prompt # Store the fully constructed prompt

    try:
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            # You can also add other parameters here if needed:
            # max_output_tokens=1024,
            # Not necessary to set, since using cumulative probability cutoff, `top_p`
            # top_k=top_k,
        )
        response = genai_model_instance.generate_content(
            final_prompt, # Use the formatted prompt
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "Sorry, I encountered an error trying to generate an answer."

# --- Streamlit UI (actual content starts here) ---
st.title("📚 RAG Application: Gemini + Phoenix Docs")
st.caption("Ask questions about Arize Phoenix, get answers from local docs, powered by Gemini.")

st.sidebar.header("Configuration")
# If model is still not configured after env and secrets, prompt in sidebar
if not model:
    api_key_input_val = st.sidebar.text_input(
        "Enter your Google API Key:",
        type="password",
        key="api_key_input_runtime",
    )
    if api_key_input_val:
        try:
            genai.configure(api_key=api_key_input_val)
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            GOOGLE_API_KEY = api_key_input_val
            st.sidebar.success("Gemini API Key configured successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to configure Gemini: {e}")
            model = None
    else:
        st.sidebar.warning("Gemini model not configured. Please enter your API key.")
elif model:
    st.sidebar.success("Gemini model configured.")

st.sidebar.header("Generation Settings")
temperature_slider = st.sidebar.slider(
    label="Model Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.3, # Default temperature
    step=0.05,
    help="""
        Controls randomness. 
        Lower values make the output more deterministic/focused, 
        higher values make it more random/creative.
        Default of 0.3 preemptively limits RAG drift.
    """,
)

top_p_slider = st.sidebar.slider(
    label="Model cumulative probability cutoff (top_p)",
    min_value=0.0,
    max_value=1.0,
    # Default top_p
    value=0.95, 
    step=0.05,
    help="""
        Influences output, post temperature.
        Limits tokens by truncating probability-ordered set in excess 
        of cumulative probability `top_p`.
        Keep around 0.95 for more precise but non-conservative Q&A.
    """,
)

st.sidebar.header("Prompt Template")
st.session_state.editable_prompt_template = st.sidebar.text_area(
    label="Edit the Prompt Template (use {rag_context} and {user_query} as placeholders):",
    value=st.session_state.editable_prompt_template,
    height=350,
    key="editable_prompt_template_widget"
)

with st.sidebar.expander("Show Last Sent Prompt to LLM", expanded=False):
    st.text_area(
        label="Last fully constructed prompt sent to Gemini:",
        value=st.session_state.last_sent_prompt,
        height=400,
        disabled=True,
        key="last_sent_prompt_display"
    )

with st.sidebar.expander("Note", expanded=False):
    st.text_area(
        label="Note",
        value=inspect.cleandoc("""
            Gemini will sometimes format the source files in <pre> tags
            when the temperature increases. 
            This also risks typically repetitive circular answers, 
            e.g., "The source file 'source_file_name.md' has the source."
        """),
        height=400,
        disabled=True,
        key="runtime_notes"
    )

# Load LlamaIndex index for Phoenix docs
phx_docs_index = load_phx_index(DOCS_PATH)

if phx_docs_index is None:
    st.warning("Phoenix documentation index could not be loaded. Context retrieval will not work.")

# Main interaction area
user_query = st.text_input("Ask your question about Phoenix Docs:", key="user_query_input")

if user_query:
    if not model:
        st.error("Gemini model is not configured. Please enter your Google API Key in the sidebar.")
    elif phx_docs_index is None:
        st.error("Phoenix Docs index is not available. Cannot retrieve context.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            retrieved_context = get_context_from_llamaindex(user_query, phx_docs_index)
            
            if "Error:" not in retrieved_context and "No relevant context found" not in retrieved_context:
                answer = generate_answer_with_rag(
                    user_query, 
                    model, 
                    retrieved_context, 
                    temperature_slider,
                    top_p_slider,
                    st.session_state.editable_prompt_template
                )
                st.subheader("Answer:")
                st.markdown(answer)
            elif "No relevant context found" in retrieved_context:
                st.info("No relevant context was found in the Phoenix Docs for your query. The LLM was not called.")
                # Optionally, you could still call the LLM without context, or with a message indicating no context
                # For this setup, we only call LLM if context is found.
            else: # An error occurred during context retrieval
                st.error(f"Could not generate an answer due to an issue with context retrieval: {retrieved_context}")

            with st.expander("Show Retrieved Context"):
                if retrieved_context:
                    st.text_area("Context from Phoenix Docs:", value=retrieved_context, height=300, disabled=True)
                else:
                    st.text("No context was retrieved or an error occurred.")
else:
    st.info("Enter a query above to search the Phoenix documentation.")

st.markdown("---")
st.markdown("This RAG application uses LlamaIndex for local document retrieval and Gemini for generation.")
st.markdown("Made as a Proof of Concept.")
