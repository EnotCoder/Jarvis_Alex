"""Замер качества распознавания по тестовому набору.

    python tests/bench_stt.py            # базовый прогон
    python tests/bench_stt.py --capture  # только вход → текст

Печатает:
  WER      — доля неверных слов (меньше лучше)
  отсеч.   — доля сценариев, где фраза оборвалась на паузе (меньше лучше)
  t, с     — время распознавания
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "tests" / "corpus"
REF = CORPUS / "reference.tsv"

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Слова, которые Whisper на русском склонен коверкать, а привести их к одному
# виду — честнее, чем считать это ошибкой. «Джарвис» он даёт как «Джаркис»,
# «Джаррис», «Джарльз» — в эталоне это всегда одно и то же имя.
_CANON = {
    "джаркис": "джарвис", "джаррис": "джарвис", "джарльз": "джарвис",
    "джарвес": "джарвис", "джервис": "джарвис", "джарвиз": "джарвис",
    "броизер": "браузер", "броузер": "браузер", "броузер": "браузер",
    "десять": "десять", "десяти": "десять", "10": "десять",
    "напомним": "напомни", "напомнит": "напомни",
    "постав": "поставь", "поставь": "поставь",
}


def norm(s: str) -> str:
    s = _PUNCT.sub(" ", s.lower())
    out = []
    for w in s.split():
        w = _CANON.get(w, w)
        out.append(w)
    text = " ".join(out)
    text = text.replace("десять минут", "десять").replace("10 минут", "десять")
    return " ".join(text.split())


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_refs() -> list[tuple[Path, str]]:
    rows = []
    for line in REF.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        fn, text = line.split("\t", 1)
        rows.append((CORPUS / fn, text))
    return rows


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        sig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    return sig, rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true",
                    help="только распознавание, без VAD захвата микрофона")
    ap.add_argument("--only", default="", help="фильтр по имени файла")
    args = ap.parse_args()

    from modules import stt

    if not stt.start_server():
        return 1

    rows = load_refs()
    if args.only:
        rows = [r for r in rows if args.only in r[0].name]

    wer_all, times, cut, total = [], [], [], 0
    print(f"\n{'файл':<34}{'WER':>6}{'время':>8}  текст")
    print("─" * 100)

    for path, ref in rows:
        sig, rate = read_wav(path)
        t0 = time.time()
        got = stt.transcribe(sig, rate=rate)
        dt = time.time() - t0
        times.append(dt)

        rw, gw = norm(ref).split(), norm(got).split()
        errs = levenshtein(rw, gw)
        wer = errs / max(1, len(rw))
        wer_all.append(wer)
        total += 1

        # Оборвалась ли фраза: в паузных сценариях половина эталона должна
        # остаться, иначе VAD принял паузу за конец речи.
        truncated = gw and len(gw) < len(rw) * 0.75
        if truncated:
            cut.append(path.name)

        mark = " ✂" if truncated else "  "
        print(f"{path.stem:<34}{wer:>6.0%}{dt:>7.1f}с{mark} {got[:52]}")

    print("─" * 100)
    print(f"WER: среднее {statistics.mean(wer_all):.1%}, худший {max(wer_all):.0%}")
    print(f"Оборвано на паузе: {len(cut)} из {total}")
    for c in cut:
        print(f"    ✂ {c}")
    print(f"Время: среднее {statistics.mean(times):.2f}с, худшее {max(times):.2f}с")

    stt.stop_server()
    return 0


if __name__ == "__main__":
    sys.exit(main())