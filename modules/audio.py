"""Аудио: захват с микрофона через PipeWire + воспроизведение.

Не тянем portaudio — работаем через pw-cat/pw-play напрямую.
Так меньше слоёв между нами и ядром, и не зависит от сборки PortAudio в Arch.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

RATE = 16000          # достаточно для Whisper
CHANNELS = 1
FRAME_MS = 30         # размер фрейма для VAD

# Порог речи считается от шумового фона, но не опускается ниже MIN_RMS:
# иначе в очень тихой комнате ловились бы потрескивания и гул.
SENSITIVITY = 2.5
MIN_RMS = 180.0

# После того как Джарвис договорил, его голос ещё долго гуляет по комнате.
# Без этой защиты микрофон ловит сам себя и ассистент начинает отвечать себе.
ECHO_GUARD = 3.0        # секунд после озвучки
ECHO_MULTIPLIER = 2.5   # во сколько раз выше порог в это время


def list_sources() -> list[tuple[int, str]]:
    """Источники записи PipeWire: [(index, name)]."""
    out = subprocess.run(["pactl", "list", "short", "sources"],
                         capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            found.append((int(parts[0]), parts[1]))
    return found


def list_sinks() -> list[tuple[int, str]]:
    """Устройства вывода PipeWire: [(index, name)]."""
    out = subprocess.run(["pactl", "list", "short", "sinks"],
                         capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            found.append((int(parts[0]), parts[1]))
    return found


def pick_source(prefer: str | None = None) -> str | None:
    """Выбирает источник записи. Исключает .monitor — иначе будет самослух и свист."""
    sources = list_sources()
    real = [(i, n) for i, n in sources if ".monitor" not in n]

    if prefer:
        for i, n in real:
            if prefer.lower() in n.lower():
                return n
        return None

    # USB-микрофон предпочтительнее встроенного: у USB-миков обычно лучше шумовой фон
    usb = [(i, n) for i, n in real if ".usb-" in n or "usb_" in n]
    pool = usb or real
    return pool[0][1] if pool else None


class MicStream:
    """Непрерывный захват с микрофона, отдаёт фреймы через очередь."""

    def __init__(self, source: str, vad_silence: float = 0.7, vad_min_speech: float = 0.55):
        # vad_min_speech: короче 0.55 с Whisper начинает додумывать —
        # на обрывках вроде «клок» он уверенно выдаёт целые слова.
        self.source = source
        self.silence_for = vad_silence
        self.min_speech = vad_min_speech
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._rms: queue.Queue[float] = queue.Queue()
        self._noise_floor = 120.0
        self._guard_until = 0.0
        self.speaking = threading.Event()

    def set_guard(self, seconds: float = ECHO_GUARD) -> None:
        """Поднимает порог на время после озвучки — защита от самослуха."""
        self._guard_until = time.monotonic() + seconds

    def start(self) -> None:
        cmd = [
            "pw-cat", "--record", "--target", self.source,
            "--rate", str(RATE), "--channels", str(CHANNELS),
            "--format", "s16", "--raw", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Читает сырой PCM из pw-cat и нарезает на фреймы."""
        frame_bytes = RATE * CHANNELS * 2 * FRAME_MS // 1000
        assert self._proc and self._proc.stdout
        buf = b""
        while True:
            chunk = self._proc.stdout.read(frame_bytes - len(buf))
            if not chunk:
                break
            buf += chunk
            if len(buf) < frame_bytes:
                continue
            samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
            self._rms.put(float(np.sqrt(np.mean(samples**2))))
            self._q.put(samples)
            buf = b""

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def clear(self) -> None:
        """Сбрасывает накопившиеся фреймы.

        После ответа Джарвиса в буфере остаётся кусок его собственной речи —
        если его не сбросить, следующий цикл услышит самого себя.
        """
        while True:
            try:
                self._q.get_nowait()
                self._rms.get_nowait()
            except queue.Empty:
                break

    def calibrate(self, seconds: float = 1.2) -> float:
        """Измеряет шумовой фон комнаты перед началом работы.

        Без этого порог пришлось бы угадывать: у USB-микрофона фон бывает
        20, у встроенного — 300, и одна константа на всех не работает.
        """
        levels: list[float] = []
        need = max(1, int(seconds * 1000 / FRAME_MS))
        for _ in range(need):
            try:
                level = self._rms.get(timeout=0.4)
                self._q.get_nowait()     # тот же фрейм, что и у level
            except queue.Empty:
                continue
            levels.append(level)

        if not levels:
            return self._noise_floor
        # Медиана, а не среднее: один щелчок мыши не должен задирать порог
        self._noise_floor = max(float(np.median(levels)), 10.0)
        return self._noise_floor

    def _threshold(self) -> float:
        """Порог речи: и над фоном, и не ниже абсолютного минимума."""
        return max(self._noise_floor * SENSITIVITY, MIN_RMS)

    def read_utterance(self, on_level=None, timeout: float = 15.0) -> np.ndarray | None:
        """Ждёт начало речи и собирает фразу целиком.

        Возвращает np-массив 16 кГц моно или None, если никто не заговорил.
        """
        started = False
        chunks: list[np.ndarray] = []
        silence_run = 0.0
        spoken = 0.0
        waited = 0.0

        while waited < timeout:
            try:
                samples = self._q.get(timeout=0.2)
            except queue.Empty:
                waited += 0.2
                if not started and waited > timeout:
                    return None
                continue

            waited += FRAME_MS / 1000
            rms = self._rms.get_nowait()
            threshold = self._threshold()
            if time.monotonic() < self._guard_until:
                threshold *= ECHO_MULTIPLIER

            if not started:
                # Фон плавно подстраивается под комнату, но только в тишине
                if rms < threshold:
                    self._noise_floor += (rms - self._noise_floor) * 0.05
                if rms > threshold:
                    started = True
                    self.speaking.set()
            else:
                spoken += FRAME_MS / 1000
                silence_run = silence_run + FRAME_MS / 1000 if rms < threshold * 0.6 else 0.0

            if started:
                chunks.append(samples)
                if on_level:
                    on_level(min(rms / 4000.0, 1.0))
                if silence_run >= self.silence_for and spoken >= self.min_speech:
                    break

        self.speaking.clear()
        if not chunks or spoken < self.min_speech:
            return None
        return np.concatenate(chunks)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int = RATE) -> np.ndarray:
    """Приводит сигнал к частоте дискретизации, которую ждёт Whisper.

    Piper отдаёт 22050 Гц, микрофон пишет 16000. Если не привести,
    фраза звучит вдвое медленнее и распознаётся как мусор.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    n_out = max(1, int(len(samples) * dst_rate / src_rate))
    x_old = np.arange(len(samples), dtype=np.float64)
    x_new = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def save_wav(samples: np.ndarray, path: str | Path, rate: int = RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def play_wav(path: str | Path, sink: str | None = None) -> None:
    """Проигрывает wav через PipeWire.

    Частоту дискретизации задавать нельзя: Piper синтезирует 22050 Гц,
    а микрофон пишет 16000. Если зафиксировать RATE, голос прозвучит
    вдвое медленнее. PipeWire сам подстроит формат под файл.
    """
    cmd = ["pw-play"]
    if sink:
        cmd += ["--target", sink]
    cmd.append(str(path))
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    print("Источники записи:")
    for i, n in list_sources():
        print(f"  [{i}] {n}")
    print("\nУстройства вывода:")
    for i, n in list_sinks():
        print(f"  [{i}] {n}")
    sel = pick_source()
    print(f"\nВыбран микрофон: {sel or 'НЕ НАЙДЕН'}")
    sys.exit(0 if sel else 1)