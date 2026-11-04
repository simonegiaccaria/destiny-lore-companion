#!/usr/bin/env bash
# Linux/macOS counterpart of avvia.bat.
#   ./avvia.sh                                   Ollama + server
#   ./avvia.sh eval --repeat 3 --temperature 0.2 Ollama + benchmark
#   ./avvia.sh ollama                            Ollama only
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[avvia] $PY not found: create the Linux virtualenv first:" >&2
    echo "        python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

exec "$PY" tools/start.py "$@"