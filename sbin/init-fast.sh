#!/bin/bash
myecho() (
  # safely color echo green
  msg="$@"  
  echo "[1;32m${msg[@]}[0m"
)

myecho "UNCOMPRESSING PHOENIX DOCS, MARKDOWN ONLY RECOVERED"
./sbin/uncompress-phx-docs-md-only
myecho "CREATING PYTHON LOCAL VIRTUAL ENVIRONMENT"
./sbin/mkpyenv
myecho "FAST INIT COMPLETE"
myecho 'Run app with: (have gemini key ready or in `etc/localdev.env`)'
myecho "./bin/run-rag-app"
