"""Генерация тестового набора для замеров распознавания.

Собирает аудио через Piper, поэтому набор воспроизводим на любой машине
и не тащит за собой чужие записи.

    python tests/make_corpus.py
"""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "corpus"
VOICE = ROOT / "models" / "tts" / "ru_RU-dmitri-medium.onnx"
SR = 22050

# Сценарии: (первая часть, [вторая часть], пауза между ними, полный эталон).
# Пауза — внутри одной реплики: так проверяется, не примет ли VAD её за конец фразы.
# Вторую часть синтезируем отдельно, иначе фраза продублируется и замер врёт.
SCENARIOS: dict[str, dict] = {
    "clean_short": {
        "parts": ["Привет, Джарвис."], "pause": 0.0,
    },
    "clean_long": {
        "parts": ["Привет, Джарвис.",
                  "Напомни, какая сегодня дата и сколько сейчас времени."],
        "pause": 0.0,
    },
    "pause_middle": {
        "parts": ["Открой, пожалуйста, браузер.",
                  "И найди в интернете рецепт борща."],
        "pause": 0.9,
    },
    "pause_short": {
        "parts": ["Запиши заметку купить молоко.",
                  "И напомни мне через час."],
        "pause": 0.5,
    },
    "pause_breath": {
        "parts": ["Поставь таймер на десять минут.",
                  "И скажи когда он закончится."],
        "pause": 0.75,
    },
    "trailing_pause": {
        "parts": ["Какая сегодня дата."], "pause": 1.4, "tail": True,
    },
}

NOISE_PROFILES = {
    "none":       None,
    "room_hum":   {"hum": 260.0, "hiss": 95.0, "clicks": 3},
    "babble":     {"hum": 90.0, "hiss": 60.0, "clicks": 2, "speech": True},
}


def synth(text: str) -> np.ndarray:
    """Синтезирует фразу голосом Piper и читает её обратно в массив."""
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_tmp.wav"
    proc = subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(VOICE), "-f", str(tmp)],
        input=text, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"Piper не отработал: {proc.stderr[-200:]}")
    with wave.open(str(tmp), "rb") as w:
        sig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    tmp.unlink(missing_ok=True)
    return sig


def join_pause(chunks: list[np.ndarray], pause: float, tail: bool = False) -> np.ndarray:
    """Склеивает куски речи паузой. tail=True — пауза только в конце."""
    if pause <= 0:
        return np.concatenate(chunks)
    gap = np.zeros(int(SR * pause), dtype=np.float32)
    if tail:
        return np.concatenate(chunks + [gap])
    out = [chunks[0]]
    for c in chunks[1:]:
        out += [gap, c]
    return np.concatenate(out)


def add_noise(sig: np.ndarray, profile: dict | None, seed: int = 7) -> np.ndarray:
    if not profile:
        return sig
    rng = np.random.default_rng(seed)
    n = len(sig)
    t = np.arange(n) / SR
    out = sig + profile.get("hum", 0) * np.sin(2 * np.pi * 50 * t)
    out = out + rng.normal(0, profile.get("hiss", 0), n)
    for _ in range(profile.get("clicks", 0)):
        pos = int(rng.integers(0, max(1, n - 400)))
        out[pos:pos + 120] += rng.normal(0, 600, 120)
    if profile.get("speech"):
        # грубая имитация чужой речи: низкочастотная пульсация + шум
        for _ in range(6):
            start = int(rng.integers(0, max(1, n - SR // 2)))
            dur = int(SR * rng.uniform(0.2, 0.45))
            tt = np.arange(dur) / SR
            env = np.sin(np.pi * np.arange(dur) / dur)
            out[start:start + dur] += env * (
                420 * np.sin(2 * np.pi * rng.uniform(90, 200) * tt)
                + rng.normal(0, 120, dur)
            )
    return out


def write(path: Path, sig: np.ndarray) -> None:
    pcm = np.clip(sig, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> int:
    if not VOICE.exists():
        print(f"Нет голоса {VOICE}. Запусти: bash setup.sh")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    refs: list[str] = []

    for name, spec in SCENARIOS.items():
        chunks = [synth(p) for p in spec["parts"]]
        base = join_pause(chunks, spec["pause"], spec.get("tail", False))
        ref = " ".join(spec["parts"])
        for noise_name, profile in NOISE_PROFILES.items():
            sig = add_noise(base, profile)
            fn = f"{name}__{noise_name}.wav"
            write(OUT / fn, sig)
            refs.append(f"{fn}\t{ref}")
        print(f"  {name}: пауза {spec['pause']}с, частей {len(chunks)}")

    (OUT / "reference.tsv").write_text("\n".join(refs), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.glob("*.wav"))
    print(f"\nГотово: {len(refs)} файлов, {total / 1e6:.1f} МБ в {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())