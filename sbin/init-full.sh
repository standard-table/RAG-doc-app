#!/bin/bash
myecho() (
  # safely color echo green
  msg="$@"  
  echo "[1;32m${msg[@]}[0m"
)
myecho "CLONING PHOENIX DOCS"
./sbin/clone-phx-docs
myecho "CREATING PYTHON LOCAL VIRTUAL ENVIRONMENT"
./sbin/mkpyenv
myecho "FULL INIT COMPLETE"
myecho 'Run app with: (have gemini key ready or in `etc/localdev.env`)'
myecho "./bin/run-rag-app"
