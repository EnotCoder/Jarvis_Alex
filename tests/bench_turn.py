"""Где ассистент решает, что фраза закончилась.

Прогоняет записи через ту же логику, что работает на живом микрофоне,
и показывает, в какой момент она решила остановиться.

    python tests/bench_turn.py

Ключевой сценарий — пауза внутри реплики. Если решено остановиться на паузе,
вторая часть фразы теряется: пользователь сказал «Открой пожалуйста браузер,
[пауза] и найди рецепт борща», а Джарвис услышал только первое.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.audio import FRAME_MS, RATE, VAD_SILENCE, Endpoint   # noqa: E402

CORPUS = ROOT / "tests" / "corpus"

# Сценарии, где пауза внутри реплики: обрезка здесь — это ошибка.
INTERNAL_PAUSE = {"pause_middle", "pause_short", "pause_breath"}



def run(path: Path, silence_for: float = VAD_SILENCE, noise_floor: float = 20.0) -> dict:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        sig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

    # Шаг кадра считаем по частоте самого файла. На микрофоне всегда 16 кГц,
    # а в тестовых записях Piper отдаёт 22 050 — при фиксированном шаге
    # счётчик времени уезжает за пределы длины записи.
    step = max(1, int(rate * FRAME_MS / 1000))
    ep = Endpoint(silence_for=silence_for)
    ep.set_noise_floor(noise_floor)

    n = 0
    for i in range(0, len(sig) - step, step):
        n += 1
        f = sig[i:i + step]
        event = ep.feed(float(np.sqrt(np.mean(f ** 2))), FRAME_MS / 1000)
        if event == "end":
            break

    captured = n * FRAME_MS / 1000
    total = len(sig) / rate
    return {
        "total": total,
        "captured": captured,
        "kept": captured / total if total else 0.0,
    }


def run_smart(path: Path, model, stride: float = 0.12,
              threshold: float = 0.5, noise_floor: float = 20.0) -> dict:
    """Прогоняет запись как поток: модель видит только то, что уже прозвучало.

    Именно так она работает в бою — решение принимается в момент, когда реплика
    ещё не закончилась, и поэтому может ошибиться: сказать «всё» на паузе.
    """
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        sig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

    # речь начинается после короткого фрагмента тишины/шума
    step = max(1, int(rate * stride))
    total = len(sig) / rate
    decision = total          # если не решил — значит дочитал до конца
    spoken = 0.0

    for end in range(step, len(sig), step):
        chunk = sig[:end]
        # грубая проверка «есть ли вообще речь», чтобы не спрашивать модель в тишине
        rms = float(np.sqrt(np.mean(chunk[-step:] ** 2)))
        if rms < max(noise_floor * 2.5, 180):
            continue
        spoken = end / rate
        if model.probability(chunk, rate) > threshold:
            decision = end / rate
            break

    return {
        "total": total,
        "captured": decision,
        "kept": decision / total if total else 0.0,
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "energy"
    model = None
    if mode == "smart":
        from modules.turn import SmartTurn
        if not SmartTurn.available():
            print("Нет модели Smart Turn. Скачай models/smart-turn-v3.2-cpu.onnx")
            return 1
        model = SmartTurn()

    runner = run if mode == "energy" else (lambda p: run_smart(p, model))
    rows = []
    for wav in sorted(CORPUS.glob("*.wav")):
        scenario, _, noise = wav.stem.partition("__")
        rows.append((scenario, noise or "none", runner(wav)))

    print(f"режим: {mode}")
    print(f"{'сценарий':<20}{'шум':<10}{'длина':>8}{'решение':>9}{'%':>7}  вердикт")
    print("─" * 74)
    bad = []
    for scenario, noise, r in rows:
        internal = scenario in INTERNAL_PAUSE
        pct = r["kept"] * 100
        if internal and pct < 90:
            verdict = "✂ ОБРЕЗАЛ НА ПАУЗЕ"
            bad.append(scenario)
        elif internal:
            verdict = "✓ дождался"
        else:
            verdict = "—"
        print(f"{scenario:<20}{noise:<10}{r['total']:>7.1f}с{r['captured']:>8.1f}с{pct:>6.0f}%  {verdict}")

    print("─" * 74)
    print(f"Срезано на внутренней паузе: {len(bad)} из {len(INTERNAL_PAUSE) * 3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())