#!/usr/bin/env bash
# JARVIS — скачивание моделей (~2.9 ГБ)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$DIR/models"
mkdir -p "$M"

HF=https://huggingface.co

get() {  # get <url> <путь>
    local url="$1" out="$2"
    if [[ -s "$out" ]]; then
        echo "  уже есть: $(basename "$out")"
        return
    fi
    echo "  качаю: $(basename "$out")"
    curl -L --fail --progress-bar -o "$out.part" "$url"
    mv "$out.part" "$out"
}

echo "==> 1/2 Whisper small (RU, ~466 МБ)"
get "$HF/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" "$M/ggml-small.bin"

echo "==> 2/2 LLM: Qwen2.5-3B-Instruct Q4_K_M (~2.2 ГБ)"
get "$HF/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
    "$M/qwen2.5-3b-instruct-q4_k_m.gguf"

echo
echo "=== Модели готовы ==="
ls -lh "$M"/*.bin "$M"/*.gguf | awk '{print "  "$9"  "$5}'