"""Smart Turn v3 — нейросетевое определение конца реплики.

Задача, которую решает энергетический порог в `audio.Endpoint`, он решает плохо:
любая пауза длиннее заданной выглядит для него как конец фразы. Из-за этого
«Открой пожалуйста браузер, [пауза] и найди рецепт борща» обрывалось на первом
куске — это видно в `tests/bench_turn.py`.

Smart Turn обучана на размеченных диалогах и отвечает на вопрос «реплика
закончилась или человек просто думает», а не «уровень звука упал».

Модель: smart-turn-v3.2-cpu.onnx, 8.3 МБ, BSD 2-Clause (Daily / Pipecat).
Работает локально на CPU, ~2 мс на вызов. Признаки считаются точно так же,
как это делает transformers.WhisperFeatureExtractor — иначе модель получит
не то, на чём обучена.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "smart-turn-v3.2-cpu.onnx"

N_SAMPLES = 16000 * 8      # модель смотрит ровно на 8 секунд
N_FFT = 400
HOP = 160
N_MELS = 80
SR = 16000
MEL_FLOOR = 1e-10
NORM_EPS = 1e-7


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    """Шкала Слэни: линейная до 1 кГц, логарифмическая выше."""
    f = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    mels = 3.0 * f / 200.0
    log_region = f >= 1000.0
    mels[log_region] = 15.0 + np.log(f[log_region] / 1000.0) * (27.0 / np.log(6.4))
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    m = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freq = 200.0 * m / 3.0
    log_region = m >= 15.0
    freq[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m[log_region] - 15.0))
    return freq


def _mel_filterbank() -> np.ndarray:
    n_bins = N_FFT // 2 + 1
    mel_min = float(_hz_to_mel(np.array([0.0]))[0])
    mel_max = float(_hz_to_mel(np.array([SR / 2.0]))[0])
    mel_freqs = np.linspace(mel_min, mel_max, N_MELS + 2)
    filter_freqs = _mel_to_hz(mel_freqs)
    fft_freqs = np.linspace(0, SR // 2, n_bins)

    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down, up))
    # Нормировка по площади треугольников, как в оригинале
    fb *= (2.0 / (filter_freqs[2:N_MELS + 2] - filter_freqs[:N_MELS]))[None, :]
    return fb


def _hann(n: int) -> np.ndarray:
    # Периодическое окно: совпадает с torch.hann_window(periodic=True)
    return np.hanning(n + 1)[:-1]


_MEL_FB = _mel_filterbank()
_WINDOW = _hann(N_FFT)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Признаки (80, 800) — ровно то, что ждёт модель."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size < N_SAMPLES:
        x = np.pad(x, (0, N_SAMPLES - x.size))
    elif x.size > N_SAMPLES:
        x = x[:N_SAMPLES]

    # Нормировка сигнала, а не спектра — так же, как в WhisperFeatureExtractor
    x = (x - x.mean()) / np.sqrt(x.var() + NORM_EPS)

    pad = N_FFT // 2
    padded = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
    windows = sliding_window_view(padded, N_FFT)[::HOP]
    spec = np.fft.rfft(windows * _WINDOW.astype(np.float64), axis=-1)
    magnitudes = (np.abs(spec) ** 2).T          # (bins, frames)

    mel = np.maximum(MEL_FLOOR, _MEL_FB.T @ magnitudes)
    log_spec = np.log10(mel)[:, :-1]            # последний кадр отбрасываем
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


class SmartTurn:
    """Оценка вероятности, что реплика закончилась."""

    def __init__(self, threshold: float = 0.5, window: float = 8.0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.inter_op_num_threads = 2
        self.session = ort.InferenceSession(str(MODEL), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.window = window

    @staticmethod
    def available() -> bool:
        return MODEL.exists()

    def probability(self, audio: np.ndarray, sample_rate: int = SR) -> float:
        """Вероятность завершения реплики по последним `window` секундам."""
        if sample_rate != SR:
            from . import audio as _a
            audio = _a.resample(audio, sample_rate, SR)
        feats = log_mel(audio[-int(SR * self.window):])
        logits = self.session.run(None, {"input_features": feats[None, ...]})[0]
        return float(1.0 / (1.0 + np.exp(-logits.reshape(-1)[0])))   # сигмоида


if __name__ == "__main__":
    # Запускаем как модуль пакета: иначе относительный импорт внутри
    # probability() не разрешится.
    import sys
    sys.path.insert(0, str(ROOT))
    import wave

    from modules.turn import SmartTurn as _ST   # noqa: E402

    if not _ST.available():
        print(f"Нет модели {MODEL}")
        raise SystemExit(1)

    st = _ST()
    for path in sys.argv[1:]:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            sig = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        p = st.probability(sig, rate)
        verdict = "✓ закончил" if p > 0.5 else "✗ ждём дальше"
        print(f"  {Path(path).stem:<30} P={p:.2f}  {verdict}")