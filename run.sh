#!/usr/bin/env bash
# JARVIS — запуск ассистента
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$DIR/.venv/bin/python" "$DIR/jarvis.py" "$@"