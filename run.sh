#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Сначала выполните: bash setup.sh'
  exit 1
fi
exec .venv/bin/python launcher.py "$@"
