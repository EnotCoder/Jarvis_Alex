"""Мозг: llama-server (локальный, GGUF). Общаемся по HTTP OpenAI-совместимого API."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import tools

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
PERSONA = ROOT / "config" / "persona.md"

HOST = "127.0.0.1"
PORT = 8099
BASE = f"http://{HOST}:{PORT}"
N_CTX = 4096
GPU_LAYERS = 99          # выгрузить всё в VRAM (6 ГБ хватает для 3B Q4)

SYSTEM = """{persona}

Сейчас {date}. Ты работаешь на компьютере Александра голосом.
Отвечай вслух, кратко и живо: это spoken-диалог, его читают на слух, а не читают глазами.

У тебя есть инструменты для действий. Правила:
- Стоит выполнить действие — вызывай инструмент, не описывай его словами.
- На «выключи» и «перезагрузи» сначала спроси подтверждение. Только после явного
  согласия вызывай инструмент снова с confirm=true.
- Результат действия уже вернётся тебе текстом. Не выдумывай, что сделал, чего не делал.
- Не перечисляй инструменты вслух, если Александр их не спрашивал.
- Отвечай строго на русском. Никаких иероглифов и английских слов, даже после
  результата действия — именно там модель чаще всего срывается на другой язык.

Приложения, которые ты умеешь запускать: {apps}."""

_server: subprocess.Popen | None = None


def _model_path() -> Path | None:
    """Берём любую GGUF, если дефолтная не скачалась."""
    if MODEL.exists():
        return MODEL
    models = sorted((ROOT / "models").glob("*.gguf"))
    return models[0] if models else None


def is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(wait: float = 90.0) -> bool:
    """Поднимает llama-server, если он ещё не работает."""
    global _server
    if is_up():
        print(f"[llm] сервер уже поднят: {BASE}")
        return True

    model = _model_path()
    if not model:
        print("[llm] модель не найдена. Запусти fetch-models.sh", file=sys.stderr)
        return False

    cmd = [
        "llama-server",
        "-m", str(model),
        "--host", HOST,
        "--port", str(PORT),
        "-c", str(N_CTX),
        "-ngl", str(GPU_LAYERS),
        "--no-webui",
    ]
    log = open(ROOT / "logs" / "llama-server.log", "w")
    _server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"[llm] запускаю llama-server на {model.name} ...")

    deadline = time.time() + wait
    while time.time() < deadline:
        if is_up():
            print("[llm] готов")
            return True
        if _server.poll() is not None:
            print("[llm] сервер упал, смотри logs/llama-server.log", file=sys.stderr)
            return False
        time.sleep(1.0)

    print("[llm] сервер не поднялся за отведённое время", file=sys.stderr)
    return False


def stop_server() -> None:
    global _server
    if _server and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server.kill()


def _system_prompt() -> str:
    persona = PERSONA.read_text(encoding="utf-8") if PERSONA.exists() else ""
    now = time.strftime("%d.%m.%Y, %H:%M")
    names = ", ".join(tools.APP_ENUM)
    return SYSTEM.format(persona=persona, date=now, apps=names)


def _post(payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(messages: list[dict], temperature: float = 0.7,
         max_tokens: int = 400, use_tools: bool = True) -> tuple[str, list[str]]:
    """Диалог с выполнением действий.

    Возвращает (текст ответа, список выполненных действий).
    Модель вызывает инструмент — мы выполняем — возвращаем результат и просим
    сформулировать ответ вслух. Максимум три круга: если модель зациклилась,
    отдаём ей то, что есть.
    """
    done: list[str] = []
    messages = list(messages)
    payload = {
        "messages": [{"role": "system", "content": _system_prompt()}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if use_tools:
        payload["tools"] = tools.SCHEMAS
        payload["tool_choice"] = "auto"

    for round_no in range(3):
        try:
            data = _post(payload)
        except urllib.error.URLError as e:
            print(f"[llm] запрос не прошёл: {e}", file=sys.stderr)
            return "", done

        message = data["choices"][0]["message"]
        calls = message.get("tool_calls")

        if not calls:
            return (message.get("content") or "").strip(), done

        payload["messages"].append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": calls,
        })

        for call_ in calls:
            fn = call_["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}

            result = tools.call(name, args)
            done.append(f"{name} → {result}")
            print(f"[tool] {name}({args}) = {result}")

            payload["messages"].append({
                "role": "tool",
                "tool_call_id": call_["id"],
                "content": result,
            })

    return (message.get("content") or "").strip() or "Готово.", done