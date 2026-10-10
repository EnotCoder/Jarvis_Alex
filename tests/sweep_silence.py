"""Подбор паузы энергетического метода: обрезает фразу или ждёт слишком долго.

    python tests/sweep_silence.py

Две величины, которые нельзя получить одновременно:
  срезано — сколько раз фраза оборвалась на паузе внутри реплики
  задержка — сколько лишнего ждём после конца речи
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.audio import FRAME_MS, Endpoint   # noqa: E402

CORPUS = ROOT / "tests" / "corpus"
INTERNAL_PAUSE = {"pause_middle", "pause_short", "pause_breath"}
NOISE_FLOOR = 20.0
THRESH = max(NOISE_FLOOR * 2.5, 180)


def analyse(path: Path) -> tuple[np.ndarray, int, float, float]:
    """Возвращает сигнал, частоту, длительность и момент конца речи."""
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        sig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    step = max(1, int(rate * FRAME_MS / 1000))

    levels = [float(np.sqrt(np.mean(sig[i:i + step] ** 2)))
              for i in range(0, len(sig) - step, step)]
    if not levels:
        return sig, rate, 0.0, 0.0

    total = len(sig) / rate
    speech_end = 0.0
    for i, v in enumerate(levels):
        if v >= THRESH:
            speech_end = (i + 1) * FRAME_MS / 1000
    return sig, rate, total, speech_end


def decide(sig: np.ndarray, rate: int, silence_for: float) -> float | None:
    """Момент, когда метод решил, что реплика закончилась."""
    step = max(1, int(rate * FRAME_MS / 1000))
    ep = Endpoint(silence_for=silence_for)
    ep.set_noise_floor(NOISE_FLOOR)
    t = 0.0
    for i in range(0, len(sig) - step, step):
        t += FRAME_MS / 1000
        rms = float(np.sqrt(np.mean(sig[i:i + step] ** 2)))
        if ep.feed(rms, FRAME_MS / 1000) == "end":
            return t
    return None


def main() -> int:
    files = sorted(CORPUS.glob("*.wav"))
    data = [(p, *analyse(p)) for p in files]

    print(f"{'пауза':>7}{'срезано':>9}{'задержка':>11}{'худшая':>9}")
    print("─" * 40)
    best = None
    for silence_for in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        cut = 0
        lats = []
        for path, sig, rate, total, speech_end in data:
            scenario = path.stem.partition("__")[0]
            t = decide(sig, rate, silence_for)
            if t is None:
                continue
            if scenario in INTERNAL_PAUSE and t < total - 0.15:
                cut += 1
            else:
                lats.append(max(0.0, t - speech_end))
        avg = sum(lats) / len(lats) if lats else 0.0
        worst = max(lats) if lats else 0.0
        flag = "  ←" if cut == 0 else ""
        print(f"{silence_for:>6.1f}с{cut:>9}{avg:>10.2f}с{worst:>8.2f}с{flag}")
        if cut == 0 and best is None:
            best = silence_for

    print("─" * 40)
    if best:
        lats = []
        for path, sig, rate, total, speech_end in data:
            scenario = path.stem.partition("__")[0]
            t = decide(sig, rate, best)
            if t is None:
                continue
            # задержку считаем только там, где решение верное: на срезанных
            # случаях t меньше конца речи и исказила бы среднее вниз
            if scenario in INTERNAL_PAUSE and t < total - 0.15:
                continue
            lats.append(max(0.0, t - speech_end))
        base = []
        for path, sig, rate, total, speech_end in data:
            scenario = path.stem.partition("__")[0]
            if scenario in INTERNAL_PAUSE:
                continue
            t = decide(sig, rate, 0.7)
            if t is not None:
                base.append(max(0.0, t - speech_end))
        print(f"Минимальная пауза без обрезаний: {best:.1f}с")
        print(f"Цена: средняя задержка {sum(lats) / len(lats):.2f}с против "
              f"{sum(base) / len(base):.2f}с при 0.7с "
              f"(+{(sum(lats) / len(lats)) - (sum(base) / len(base)):.2f}с к отклику)")
    else:
        print("Ни одна пауза не убирает обрезания полностью")
    return 0


if __name__ == "__main__":
    sys.exit(main())