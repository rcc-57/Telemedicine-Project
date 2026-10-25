#!/bin/bash
set -e
cd "$(dirname "$0")"
echo 'Окно врача: нажмите «Я врач • создать» и скопируйте код комнаты.'
echo 'Во втором терминале запустите: bash run-patient.sh'
exec bash run.sh --bind 127.0.0.1 --host 127.0.0.1 "$@"
