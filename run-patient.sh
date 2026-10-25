#!/bin/bash
set -e
cd "$(dirname "$0")"
echo 'Окно пациента: введите код из окна врача и нажмите «Я пациент • войти».'
echo 'Сначала запустите bash run-doctor.sh в первом терминале.'
exec bash run.sh --client-only --host 127.0.0.1 "$@"
