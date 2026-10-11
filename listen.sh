#!/usr/bin/env bash
# Прослушивание голосов: ./listen.sh dmitri | ./listen.sh silero | ./listen.sh all
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs/voice_samples"
WHO="${1:-all}"

SINK=$(pactl get-default-sink 2>/dev/null || echo "")
play() { [ -n "$SINK" ] && pw-play --target "$SINK" "$1" >/dev/null 2>&1 || pw-play "$1" >/dev/null 2>&1; }

if [ "$WHO" = "all" ]; then
    for v in dmitri denis ruslan irina silero; do
        echo "=== $v ==="
        for f in "$DIR/${v}"_*.wav; do
            [ -e "$f" ] || continue
            echo "    $(basename "$f")"
            play "$f"
        done
    done
else
    for f in "$DIR/${WHO}"_*.wav; do
        [ -e "$f" ] || { echo "нет файлов для $WHO"; exit 1; }
        echo "    $(basename "$f")"
        play "$f"
    done
fi
echo "готово"