#!/usr/bin/env bash
# JARVIS — установка системных пакетов
# Запуск:  sudo bash ~/Vulkan/jarvis/install.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$DIR/models"

if [[ $EUID -ne 0 ]]; then
    echo "Нужен root. Запусти: sudo bash $0" >&2
    exit 1
fi

echo "==> Ставлю пакеты Arch"
pacman -S --needed --noconfirm \
    llama-cpp \
    whisper-cpp \
    ggml-vulkan \
    vulkan-tools \
    ffmpeg

echo
echo "==> Проверяю CUDA-свободный Vulkan-бэкенд"
ls /usr/lib/ggml/ 2>/dev/null || echo "ВНИМАНИЕ: /usr/lib/ggml пуст — GPU-бэкенд не установлен"

echo
echo "==> Проверяю, что видеокарта видна через Vulkan"
if command -v vulkaninfo >/dev/null; then
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" || echo "ВНИМАНИЕ: Vulkan не видит GPU"
else
    echo "vulkaninfo не найден"
fi

echo
echo "==> Готовлю каталог моделей: $MODELS"
mkdir -p "$MODELS"

echo
echo "=== Системные пакеты готовы ==="
echo "Дальше: bash $DIR/fetch-models.sh"