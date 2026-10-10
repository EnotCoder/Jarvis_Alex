#!/usr/bin/env python3
"""JARVIS — локальный голосовой ассистент.

    Микрофон → Whisper → Qwen (локально) → Piper → колонки

Запуск:  ./run.sh
"""
from __future__ import annotations

import signal
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules import audio, llm, stt, tools, tts

ROOT = Path(__file__).resolve().parent
VOICE = "dmitri"

# Цвета
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YEL = "\033[33m"
C_RED = "\033[31m"
C_OFF = "\033[0m"

HISTORY_LIMIT = 12          # пар реплик в памяти


class HUD:
    """Минималистичный индикатор состояния в терминале."""

    def __init__(self):
        self.state = "boot"

    def show(self, state: str, detail: str = "") -> None:
        marks = {
            "boot":   (C_DIM,  "· инициализация"),
            "listen": (C_CYAN, "● слушаю"),
            "think":  (C_YEL,  "◐ думаю"),
            "speak":  (C_GREEN,"◆ говорю"),
            "error":  (C_RED,  "✖ ошибка"),
        }
        color, label = marks.get(state, (C_DIM, state))
        sys.stdout.write(f"\r{C_DIM}╭─{C_OFF} {color}{label:<16}{C_OFF} {C_DIM}{detail[:46]:<46}{C_OFF} ")
        sys.stdout.flush()

    def line(self, text: str, color: str = C_DIM) -> None:
        print(f"{C_DIM}│{C_OFF} {color}{text}{C_OFF}")


def say(hud: HUD, text: str, voice: str = VOICE) -> None:
    hud.show("think", "синтез речи")
    wav = tts.speak(text, voice=voice)
    if wav:
        hud.show("speak", text[:46])
        sink = audio.list_sinks()[0][1] if audio.list_sinks() else None
        audio.play_wav(wav, sink=sink)
    else:
        hud.line("не удалось синтезировать речь", C_RED)


def preflight(hud: HUD, need_mic: bool = True) -> bool:
    """Проверяет всё до старта, чтобы не падать посреди разговора."""
    ok = True

    # В режиме проверки файла микрофон не нужен: аудио уже лежит на диске.
    mic = audio.pick_source() if need_mic else None
    if need_mic and not mic:
        hud.show("error", "микрофон не найден")
        print(f"{C_RED}Микрофон не найден.{C_OFF} Проверь USB и PulseAudio/PipeWire.")
        ok = False
    elif mic:
        hud.line(f"микрофон: {mic.split('.')[-2] if '.' in mic else mic}")

    if not stt.available():
        hud.show("error", "Whisper не установлен")
        print(f"{C_RED}whisper-server нет или нет модели.{C_OFF}")
        print("  sudo bash install.sh && bash fetch-models.sh")
        ok = False
    elif not stt.start_server():
        print(f"{C_RED}Whisper не поднялся, смотри logs/whisper-server.log{C_OFF}")
        ok = False

    if not tts.available(VOICE):
        hud.show("error", "голос Piper не найден")
        print(f"{C_RED}Нет голоса Piper.{C_OFF} Скачай ru_RU-dmitri-medium.onnx в models/tts/")
        ok = False

    return ok


def run_once(hud: HUD, heard: str, history: list[dict]) -> None:
    """Один полный оборот: текст → действия → голос."""
    if not heard or len(heard) < 2:
        hud.line("не разобрал", C_YEL)
        return

    hud.line(f"вы: {heard}", C_CYAN)
    history.append({"role": "user", "content": heard})

    t0 = time.time()
    reply, actions = llm.chat(history[-HISTORY_LIMIT:])
    if not reply:
        hud.line("мозг не ответил", C_RED)
        return
    history.append({"role": "assistant", "content": reply})

    for act in actions:
        hud.line(f"  ⚙ {act}", C_DIM)
    hud.line(f"джарвис ({time.time() - t0:.1f}с): {reply}", C_GREEN)
    say(hud, reply)


def main() -> int:
    hud = HUD()
    hud.show("boot", "проверка системы")
    print(f"{C_CYAN}╔══════════════════════════════════════════╗{C_OFF}")
    print(f"{C_CYAN}║{C_OFF}  JARVIS v0.1 — локальная версия       {C_CYAN}║{C_OFF}")
    print(f"{C_CYAN}╚══════════════════════════════════════════╝{C_OFF}\n")

    if not preflight(hud, need_mic="--file" not in sys.argv):
        return 1

    if not llm.start_server():
        return 1

    history: list[dict] = []

    # Режим проверки: ./run.sh --file реплика.wav — без микрофона, один оборот.
    # Нужен, когда не слышно, а проверить цепочку надо.
    if "--file" in sys.argv:
        path = sys.argv[sys.argv.index("--file") + 1]
        import wave
        w = wave.open(path)
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        hud.show("think", "распознаю")
        run_once(hud, stt.transcribe(raw, rate=w.getframerate()), history)
        return 0

    mic_name = audio.pick_source()
    mic = audio.MicStream(mic_name)
    mic.start()
    time.sleep(0.8)      # даём pw-cat раскачаться

    hud.show("boot", "калибрую микрофон")
    floor = mic.calibrate(seconds=1.5)
    hud.line(f"шумовой фон: {floor:.0f} → порог речи: {max(floor * 2.5, 180):.0f}\n")

    # Таймеры должны уметь заговорить вслух посреди разговора
    tools.set_say_callback(lambda text: say(hud, text))

    hud.line("готов. говорите — я слушаю. Ctrl+C — выход.\n")

    try:
        while True:
            hud.show("listen", "жду речи")

            audio_chunk = mic.read_utterance(on_level=lambda lv: None)
            if audio_chunk is None:
                continue

            hud.show("think", "распознаю")
            run_once(hud, stt.transcribe(audio_chunk), history)

            # Колонки рядом с микрофоном: пока звук не затух, поднимаем порог,
            # иначе Джарвис услышит сам себя и начнёт отвечать на свой же голос
            mic.set_guard()
            mic.clear()
            time.sleep(0.4)
            mic.clear()

    except KeyboardInterrupt:
        print(f"\n{C_DIM}выключаюсь...{C_OFF}")
    finally:
        mic.stop()
        stt.stop_server()
        llm.stop_server()
    return 0


if __name__ == "__main__":
    # kill вместо Ctrl+C иначе обрывает процесс без блока finally,
    # и временные файлы остаются на диске
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.exit(main())