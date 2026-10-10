"""Синтез речи: Piper (onnxruntime), локально, без интернета."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "models" / "tts"
DEFAULT_VOICE = "dmitri"          # dmitri | denis | ruslan | irina


def voice_path(name: str = DEFAULT_VOICE) -> Path:
    return VOICES_DIR / f"ru_RU-{name}-medium.onnx"


def available(voice: str = DEFAULT_VOICE) -> bool:
    return voice_path(voice).exists()


def speak(text: str, voice: str = DEFAULT_VOICE,
          length_scale: float = 1.0, volume: float = 1.0) -> Path | None:
    """Синтезирует речь и возвращает путь к wav. None — если нечего озвучивать."""
    text = text.strip()
    if not text:
        return None

    model = voice_path(voice)
    if not model.exists():
        raise FileNotFoundError(f"Голос не найден: {model}")

    out = Path("/tmp/jarvis-tts")
    out.mkdir(exist_ok=True)
    wav = out / "out.wav"

    # Piper ожидает текст на stdin
    proc = subprocess.run(
        [sys.executable, "-m", "piper",
         "-m", str(model),
         "-f", str(wav),
         "--length-scale", str(length_scale),
         "--volume", str(volume)],
        input=text, capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0 or not wav.exists():
        print(f"[tts] ошибка: {proc.stderr[-300:]}", file=sys.stderr)
        return None
    return wav


if __name__ == "__main__":
    import sys as _s
    v = _s.argv[1] if len(_s.argv) > 1 else DEFAULT_VOICE
    p = speak("Добрый вечер, Александр. Я готов к работе.", voice=v)
    print("озвучено:", p)