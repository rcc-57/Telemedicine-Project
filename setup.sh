#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  if command -v python3.11 >/dev/null; then
    python3.11 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements-desktop.txt
.venv/bin/python scripts/setup_models.py
.venv/bin/python launcher.py --download-model
