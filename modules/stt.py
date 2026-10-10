"""Распознавание речи: whisper.cpp в режиме сервера.

Два решения, проверенные на практике:

1. Модель держится в памяти (whisper-server), а не грузится на каждый запрос.
   Через whisper-cli каждая фраза стоила ~20 секунд, из которых 13 — загрузка
   487 МБ модели с диска.

2. Распознавание идёт на CPU (-ng), а не на GPU. Бэкенд ggml-vulkan на драйвере
   NVIDIA для энкодера Whisper медленнее CPU примерно в десять раз:
   5223 мс против 535 мс на этой машине. GPU в этой сборке отдан LLM — там
   он даёт выигрыш, которого Whisper не даёт.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from . import audio

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "ggml-small.bin"
VAD_MODEL = ROOT / "models" / "ggml-silero-v5.1.2.bin"
LOG = ROOT / "logs" / "whisper-server.log"

HOST = "127.0.0.1"
PORT = 8081
BASE = f"http://{HOST}:{PORT}"
THREADS = 6
LANG = "ru"

# Подсказку декодеру (--prompt) проверили и убрали: выигрыша на русском не даёт,
# а в одном из тестов подтолкнула Whisper вставить слово «таймер» в фразу,
# где его не было. Лучше пустой промпт, чем подсказка со списком слов.

_server: subprocess.Popen | None = None
_workdir: str | None = None


def available() -> bool:
    from shutil import which
    return _model_path() is not None and which("whisper-server") is not None


def _model_path() -> Path | None:
    if MODEL.exists():
        return MODEL
    models = sorted((ROOT / "models").glob("ggml-*.bin"))
    return models[0] if models else None


def is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(wait: float = 60.0) -> bool:
    """Поднимает whisper-server, если он ещё не работает."""
    global _server
    if is_up():
        print(f"[stt] сервер уже поднят: {BASE}")
        return True

    model = _model_path()
    if not model:
        print("[stt] модель не найдена. Запусти fetch-models.sh")
        return False

    cmd = [
        "whisper-server",
        "-m", str(model),
        "-l", LANG,
        "-t", str(THREADS),
        "-ng",              # CPU: см. докстринг модуля
        "--host", HOST,
        "--port", str(PORT),
        "--convert",        # отдавать JSON, а не сырой текст
    ]

    # VAD отсекает тишину и шум до того, как они попадут в декодер.
    # Без него Whisper на пустом сигнале выдаёт «Редактор субтитров
    # Н.Закомолдина», а на шуме — «звук птиц». Проверено на этой машине.
    if VAD_MODEL.exists():
        cmd += ["--vad", "--vad-model", str(VAD_MODEL),
                "--vad-threshold", "0.5", "--vad-min-speech-duration-ms", "250"]
    else:
        print("[stt] нет модели VAD — на шуме будут появляться выдуманные фразы")
    LOG.parent.mkdir(exist_ok=True)
    log = open(LOG, "w")
    # whisper-server сохраняет каждый загруженный файл в текущем каталоге —
    # без cwd он насыпает огрызки в папку проекта.
    workdir = tempfile.mkdtemp(prefix="jarvis-whisper-")
    globals()["_workdir"] = workdir
    _server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=workdir)
    print(f"[stt] запускаю whisper-server на {model.name} (CPU) ...")

    deadline = time.time() + wait
    while time.time() < deadline:
        if is_up():
            print("[stt] готов")
            return True
        if _server.poll() is not None:
            print(f"[stt] сервер упал, смотри {LOG}")
            return False
        time.sleep(0.5)

    print("[stt] сервер не поднялся вовремя")
    return False


def stop_server() -> None:
    global _server, _workdir
    if _server and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server.kill()
    if _workdir:
        shutil.rmtree(_workdir, ignore_errors=True)
        _workdir = None


def _clean(text: str) -> str:
    """Убирает служебные метки, которые Whisper любит вставлять."""
    text = re.sub(r"\[[^\]]*\]", "", text)      # [BLANK_AUDIO], [MUSIC]
    text = text.replace("**", "").replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def transcribe(samples: np.ndarray, lang: str = LANG, rate: int = audio.RATE) -> str:
    """Отправляет аудио в Whisper и возвращает распознанный текст.

    rate — частота дискретизации входных данных. Если она не 16 кГц,
    сигнал пересэмплируется: иначе фраза звучит в разы медленнее.
    """
    samples = audio.resample(samples, rate, audio.RATE)
    wav = audio.save_wav(samples, Path("/tmp/jarvis-stt") / f"{uuid.uuid4().hex}.wav")

    boundary = f"----jarvis{uuid.uuid4().hex}"
    body = bytearray()
    # Внимание: поле "file" здесь быть не должно. Если отправить его текстовым
    # полем раньше файла, сервер возьмёт именно его и попытается открыть строку
    # с именем как будто это аудио — отсюда "Invalid data found when processing input".
    for field, value in (("temperature", "0.0"), ("language", lang),
                         ("response_format", "json")):
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{field}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'.encode()
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += wav.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE}/inference", data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"[stt] запрос не прошёл: {e}")
        return ""
    finally:
        wav.unlink(missing_ok=True)

    return _clean(data.get("text", ""))