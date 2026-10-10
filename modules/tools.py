"""Действия: всё, что Джарвис умеет делать руками.

Здесь нет и не будет `shell=True` с произвольной строкой от модели.
Набор команд зашит белым списком: модель выбирает только из списка имён.
Причина практическая — на 3B модели любая возможность «выполнить строку»
рано или поздно приведёт к тому, что фраза вида «удали все файлы, которые
начинаются на...» будет выполнена буквально.

Всё запускается в фоне через Popen, чтобы диалог не подвисал.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = Path.home() / "Pictures" / "jarvis"
NOTES = ROOT / "notes.md"

_say: callable | None = None      # куда отправлять озвучку (проставляет jarvis.py)
_timers: list[threading.Timer] = []


def set_say_callback(fn) -> None:
    _say = fn


# ──────────────────────────────────────────────────────────── утилиты

def _run(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    """Запускает команду и возвращает (успех, вывод). Никогда не бросает."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or p.stderr).strip()
        return p.returncode == 0, out
    except FileNotFoundError:
        return False, f"команда не найдена: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, "команда не ответила вовремя"


def _launch(cmd: list[str]) -> bool:
    """Запускает приложение в фоне, не дожидаясь его закрытия."""
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return True
    except FileNotFoundError:
        return False


# ──────────────────────────────────────────────────────────── приложения

APPS: dict[str, list[str]] = {
    "firefox":    ["firefox"],
    "browser":    ["firefox"],
    "terminal":   ["alacritty"],
    "files":      ["nautilus"],
    "calculator": ["galculator"],
    "gimp":       ["gimp"],
    "imageviewer": ["eog"],
}
# Синонимы, которые может выдать голос или модель
APP_ALIASES = {
    "фаерфокс": "firefox", "файрфокс": "firefox", "браузер": "firefox",
    "интернет": "firefox", "интернет-браузер": "firefox", "firefoks": "firefox",
    "терминал": "terminal", "консоль": "terminal", "терминаль": "terminal",
    "командная строка": "terminal", "shell": "terminal",
    "проводник": "files", "файлы": "files", "файловый менеджер": "files", "папки": "files",
    "калькулятор": "calculator", "кальк": "calculator", "счётчик": "calculator",
    "гмп": "gimp", "джимп": "gimp", "фотошоп": "gimp", "редактор изображений": "gimp",
    "просмотр фото": "imageviewer",
}


def open_app(app: str = "", query: str = "") -> str:
    key = APP_ALIASES.get(app.strip().lower(), app.strip().lower())
    cmd = APPS.get(key)
    if not cmd:
        return f"Приложение «{app}» мне неизвестно. Умею: {', '.join(APPS)}"

    full = cmd + ([query] if query else [])
    if not _launch(full):
        return f"Не удалось запустить {key}: команда не найдена."
    if query:
        return f"Открыл {key} с запросом «{query}»."
    return f"Открыл {key}."


def open_url(url: str = "") -> str:
    url = url.strip()
    if not url:
        return "Пустой адрес."
    if not re.match(r"^https?://", url):
        if "." not in url:
            return f"«{url}» не похоже на адрес сайта."
        url = "https://" + url
    return ("Открыл сайт." if _launch(["xdg-open", url])
            else "Не удалось открыть сайт.")


def web_search(query: str = "") -> str:
    if not query.strip():
        return "Пустой поисковый запрос."
    import urllib.parse
    url = "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
    return ("Ищу в интернете." if _launch(["xdg-open", url])
            else "Поиск не удался запустить.")


# ──────────────────────────────────────────────────────────── система

def system_info(what: str = "time") -> str:
    what = what.strip().lower()
    now = datetime.now()

    if what in ("time", "время"):
        return f"Сейчас {now:%H:%M:%S}."
    if what in ("date", "дата"):
        return f"Сегодня {now:%d} {now:%B} {now:%Y} года, {now:%A}."
    if what in ("uptime", "аптайм"):
        ok, out = _run(["uptime", "-p"])
        return f"Система работает {out[3:]}." if ok else "Не удалось узнать время работы."
    if what in ("cpu", "процессор"):
        ok, out = _run(["uptime"])
        return f"Загрузка: {out.split('load average:')[-1].strip()}." if ok else "Нет данных."
    if what in ("memory", "ram", "память", "оперативная память"):
        ok, out = _run(["free", "-h"])
        used = total = "?"
        for line in out.splitlines():
            if line.startswith("Mem:"):
                f = line.split()
                total, used = f[1], f[2]
        return (f"Оперативная память: занято {used} из {total}."
                if ok else "Нет данных о памяти.")
    if what in ("gpu", "видеокарта"):
        ok, out = _run(["nvidia-smi",
                        "--query-gpu=name,memory.used,memory.total",
                        "--format=csv,noheader"])
        return out if ok else "Информация о видеокарте недоступна."
    if what in ("disk", "диск", "место"):
        ok, out = _run(["df", "-h", str(Path.home())])
        line = out.splitlines()[1] if ok and len(out.splitlines()) > 1 else ""
        parts = line.split()
        return (f"Диск: занято {parts[4]} из {parts[1]} ({parts[5]})."
                if len(parts) > 5 else "Нет данных о диске.")
    return "Не знаю, что именно посмотреть: время, дата, память, диск, видеокарта, загрузка."


def volume(action: str = "get", level: int | None = None) -> str:
    sink = "@DEFAULT_SINK@"
    action = action.strip().lower()

    if action in ("get", "уровень", "текущий"):
        ok, out = _run(["pactl", "get-sink-volume", sink])
        if not ok:
            return "Не удалось узнать громкость."
        try:
            pct = out.split("/")[1].strip().rstrip("%")
        except IndexError:
            return f"Громкость: {out}"
        mute_ok, mout = _run(["pactl", "get-sink-mute", sink])
        mute = " (выключен)" if mute_ok and "yes" in mout else ""
        return f"Громкость {pct}%{mute}."

    if action in ("mute", "выключить", "заглушить"):
        _run(["pactl", "set-sink-mute", sink, "1"])
        return "Звук выключен."
    if action in ("unmute", "включить", "включить звук"):
        _run(["pactl", "set-sink-mute", sink, "0"])
        return "Звук включён."

    delta = {"up": 10, "громче": 10, " louder": 10, "down": -10,
             "тише": -10, "тише на": -10, "убавь": -10}.get(action, None)
    if action in ("set", "установить"):
        target = max(0, min(100, int(level or 50)))
        _run(["pactl", "set-sink-volume", sink, f"{target}%"])
        return f"Громкость {target}%."
    if delta is not None:
        ok, out = _run(["pactl", "get-sink-volume", sink])
        if not ok:
            return "Не удалось изменить громкость."
        try:
            cur = int(out.split("/")[1].strip().rstrip("%").strip())
        except (IndexError, ValueError):
            return "Не удалось разобрать текущую громкость."
        target = max(0, min(100, cur + delta))
        _run(["pactl", "set-sink-volume", sink, f"{target}%"])
        return f"Громкость {cur}%. Стало {target}%."
    return "Не понял, что сделать с громкостью."


def _session() -> str:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() or ("wayland" if os.environ.get("WAYLAND_DISPLAY") else "x11")


def screenshot(area: str = "screen") -> str:
    """Снимок экрана, окна или выделенной области.

    На X11 `grim` не работает в принципе — он только для Wayland,
    поэтому инструмент подбирает утилиту под текущую сессию.
    """
    SHOTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = SHOTS / f"{stamp}.png"
    area = area.strip().lower()
    session = _session()

    if area in ("area", "область", "выделить"):
        if session == "wayland":
            try:
                geom = subprocess.run(["slurp"], capture_output=True,
                                      text=True, timeout=30).stdout.strip()
            except subprocess.TimeoutExpired:
                return "Выделение области отменено."
            if not geom:
                return "Область не выбрана."
            cmd, what = ["grim", "-g", geom, str(path)], "снимок выделенной области"
        else:
            ok, _ = _run(["maim", "-s", str(path)], timeout=60)
            if ok:
                return f"Готово: снимок выделенной области в {path}."
            return "Область не выбрана или снимок не удался."

    elif area in ("window", "окно"):
        if session == "wayland":
            cmd, what = ["grim", "-w", str(path)], "снимок окна"
        else:
            active = subprocess.run(["xdotool", "getactivewindow"],
                                    capture_output=True, text=True).stdout.strip()
            if not active:
                return "Не удалось определить активное окно."
            cmd = ["import", "-window", active, str(path)]
            what = "снимок окна"

    else:
        if session == "wayland":
            cmd, what = ["grim", str(path)], "снимок экрана"
        else:
            cmd, what = ["import", "-window", "root", str(path)], "снимок экрана"

    ok, _ = _run(cmd, timeout=30)
    return f"Готово: {what} в {path}." if ok else "Не удалось сделать снимок."


def notify(message: str = "") -> str:
    if not message.strip():
        return "Пустое уведомление."
    _launch(["notify-send", "Джарвис", message])
    return "Уведомление отправлено."


def windows(action: str = "list") -> str:
    ok, out = _run(["wmctrl", "-l"])
    if not ok:
        return "Не удалось получить список окон."
    rows = [l.split(None, 3) for l in out.splitlines() if l.strip()]
    action = action.strip().lower()

    if action in ("list", "список"):
        if not rows:
            return "Открытых окон нет."
        # Номера и заголовки обрезаем: это пойдёт в синтез речи, а не на экран
        items = [f"{i + 1}. {r[3].strip()[:28]}" for i, r in enumerate(rows[:6])]
        more = f" и ещё {len(rows) - 6}" if len(rows) > 6 else ""
        return "Открыто " + str(len(rows)) + " окон: " + "; ".join(items) + more + "."

    m = re.search(r"(\d+)\s*$", action)
    if action in ("close", "закрыть") and m:
        idx = int(m.group(1)) - 1
        if not 0 <= idx < len(rows):
            return f"Окна {m.group(1)} нет."
        title = rows[idx][3].strip()[:30]
        _run(["wmctrl", "-i", "-c", rows[idx][0]])
        return f"Закрыл окно «{title}»."

    if action in ("focus", "на передний план", "переключись") and m:
        idx = int(m.group(1)) - 1
        if not 0 <= idx < len(rows):
            return f"Окна {m.group(1)} нет."
        _run(["wmctrl", "-i", "-a", rows[idx][0]])
        return f"Переключился на «{rows[idx][3].strip()[:30]}»."

    return "Назови номер окна: например «закрой окно 2»."


def list_apps() -> str:
    """Что Джарвис умеет запускать."""
    ready = [k for k, v in APPS.items() if shutil.which(v[0])]
    missing = [k for k in APPS if k not in ready]
    out = "Умею запускать: " + ", ".join(ready) + "."
    if missing:
        out += " Не установлены: " + ", ".join(missing) + "."
    return out


def power(action: str = "", confirm: bool = False) -> str:
    action = action.strip().lower()
    if not action:
        return "Что сделать с питанием: заблокировать, сон, выключить, перезагрузить?"

    if action in ("lock", "заблокировать", "залочить"):
        ok, _ = _run(["loginctl", "lock-session"])
        return "Экран заблокирован." if ok else "Не удалось заблокировать."
    if action in ("suspend", "сон", "спать"):
        return "Усыпляю. Пока." if _launch(["systemctl", "suspend"]) else "Не удалось усыпить."

    # Действия с потерей данных требуют явного подтверждения.
    if action in ("shutdown", "выключить", "выключение", "погасить"):
        if not confirm:
            return "Выключить машину? Мне нужно подтверждение."
        threading.Timer(5.0, lambda: _launch(["systemctl", "poweroff"])).start()
        return "Выключаюсь. Пока."
    if action in ("reboot", "перезагрузить", "перезагрузка", "ребутануть"):
        if not confirm:
            return "Перезагрузить машину? Мне нужно подтверждение."
        threading.Timer(5.0, lambda: _launch(["systemctl", "reboot"])).start()
        return "Перезагружаюсь. Пока."
    return "Не понял: заблокировать, сон, выключить или перезагрузить?"


def add_note(text: str = "") -> str:
    if not text.strip():
        return "Пустая заметка."
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(f"- [{stamp}] {text.strip()}\n")
    return f"Записал в заметки: {text.strip()[:70]}"


def read_notes() -> str:
    if not NOTES.exists():
        return "Заметок пока нет."
    lines = [l.strip("- \n") for l in NOTES.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return "Заметок пока нет."
    return "Последние заметки: " + "; ".join(lines[-5:]) + "."


def set_timer(seconds: int = 300, what: str = "") -> str:
    """Ставит таймер и через N секунд произносит текст вслух."""
    seconds = max(1, min(86400, int(seconds or 0)))
    if seconds == 0:
        return "Сколько секунд?"
    what = what.strip() or f"{seconds} секунд прошло"

    def fire():
        if _say:
            _say(f"Александр, {what}.")
        else:
            print(f"[таймер] {what}")

    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()
    _timers.append(t)

    if seconds >= 3600:
        left = f"{seconds // 3600} ч {(seconds % 3600) // 60} мин"
    elif seconds >= 60:
        left = f"{seconds // 60} мин {seconds % 60} сек"
    else:
        left = f"{seconds} сек"
    return f"Таймер поставлен на {left}."


def cancel_timers() -> str:
    n = 0
    for t in _timers:
        if t.is_alive():
            t.cancel()
            n += 1
    _timers.clear()
    return f"Отменил таймеров: {n}." if n else "Активных таймеров нет."


# ──────────────────────────────────────────────────────────── реестр

def _s(name, desc, props, required=None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": required or []}}}

APP_ENUM = list(APPS)

SCHEMAS: list[dict] = [
    _s("open_app", "Открыть приложение на компьютере. Используй для просьб вида «открой браузер», «запусти терминал».",
       {"app": {"type": "string", "enum": APP_ENUM, "description": "Приложение"},
        "query": {"type": "string", "description": "Что открыть: поисковый запрос, адрес или путь к файлу. Пусто — просто открыть."}},
       ["app"]),
    _s("open_url", "Открыть конкретный сайт по адресу.",
       {"url": {"type": "string", "description": "Адрес сайта"}}, ["url"]),
    _s("web_search", "Искать информацию в интернете и показать результаты в браузере.",
       {"query": {"type": "string", "description": "Поисковый запрос"}}, ["query"]),
    _s("system_info", "Узнать сведения о системе: время, дата, загрузка процессора, память, диск, видеокарта, время работы.",
       {"what": {"type": "string", "enum": ["time", "date", "uptime", "cpu", "memory", "gpu", "disk"],
                 "description": "Что именно посмотреть"}}, ["what"]),
    _s("volume", "Управление громкостью звука.",
       {"action": {"type": "string", "enum": ["get", "up", "down", "mute", "unmute", "set"],
                   "description": "Что сделать"},
        "level": {"type": "integer", "minimum": 0, "maximum": 100,
                  "description": "Уровень для действия set"}}, ["action"]),
    _s("screenshot", "Сделать снимок экрана. Для области нужно выделить её мышью.",
       {"area": {"type": "string", "enum": ["screen", "window", "area"], "description": "Область съёмки"}},
       ["area"]),
    _s("notify", "Показать уведомление на экране.",
       {"message": {"type": "string", "description": "Текст уведомления"}}, ["message"]),
    _s("windows", "Список открытых окон, переключиться на окно или закрыть его.",
       {"action": {"type": "string", "enum": ["list", "focus", "close"],
                   "description": "Действие; для focus и close — номер окна из списка, например «2»"}},
       ["action"]),
    _s("list_apps", "Какие приложения ассистент умеет запускать.", {}),
    _s("power", "Управление питанием: блокировка, сон, выключение, перезагрузка. Для выключения и перезагрузки нужен confirm=true.",
       {"action": {"type": "string", "enum": ["lock", "suspend", "shutdown", "reboot"], "description": "Действие"},
        "confirm": {"type": "boolean", "description": "Подтверждение от пользователя. Ставить true только если он явно согласился."}},
       ["action"]),
    _s("add_note", "Записать заметку в файл, чтобы не забыть.",
       {"text": {"type": "string", "description": "Текст заметки"}}, ["text"]),
    _s("read_notes", "Прочитать последние заметки.", {}),
    _s("set_timer", "Поставить таймер и через указанное время произнести текст вслух.",
       {"seconds": {"type": "integer", "minimum": 1, "description": "Через сколько секунд"},
        "what": {"type": "string", "description": "Что произнести, например «чайник закипел»"}},
       ["seconds"]),
    _s("cancel_timers", "Отменить все активные таймеры.", {}),
]

HANDLERS = {
    "open_app": open_app, "open_url": open_url, "web_search": web_search,
    "system_info": system_info, "volume": volume, "screenshot": screenshot,
    "notify": notify, "windows": windows, "power": power,
    "add_note": add_note, "read_notes": read_notes,
    "set_timer": set_timer, "cancel_timers": cancel_timers,
    "list_apps": list_apps,
}


def call(name: str, args: dict) -> str:
    """Выполняет инструмент по имени. Никогда не бросает — возвращает текст."""
    fn = HANDLERS.get(name)
    if not fn:
        return f"Инструмента «{name}» нет."
    clean = {k: v for k, v in (args or {}).items() if v is not None}
    try:
        return fn(**clean) or "Готово."
    except TypeError as e:
        return f"Неверные аргументы для {name}: {e}"
    except Exception as e:                       # noqa: BLE001 — диалог не должен падать
        return f"Действие {name} не выполнилось: {e}"


if __name__ == "__main__":
    print(f"инструментов: {len(SCHEMAS)}")
    for s in SCHEMAS:
        print(f"  {s['function']['name']}")
    print("\nпроверка:")
    print(" ", call("system_info", {"what": "time"}))
    print(" ", call("volume", {"action": "get"}))
    print(" ", call("windows", {"action": "list"}))
    print(" ", call("power", {"action": "shutdown"}))