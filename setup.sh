#!/usr/bin/env bash
# JARVIS — окружение Python
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$DIR/.venv" ]]; then
    echo "==> Создаю venv"
    python3 -m venv "$DIR/.venv"
    "$DIR/.venv/bin/pip" install -q --upgrade pip
fi

echo "==> Ставлю зависимости"
"$DIR/.venv/bin/pip" install -q numpy piper-tts

echo "==> Готово"
"$DIR/.venv/bin/python" -c "import numpy, piper; print('numpy', numpy.__version__, '| piper ok')"