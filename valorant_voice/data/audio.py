"""Pure audio helpers: no files, no models, so they are cheap to test."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

TARGET_RMS_DBFS = -23.0
SILENCE_DBFS = -45.0
FRAME_SECONDS = 0.02


def to_mono(samples: np.ndarray) -> np.ndarray:
    return samples if samples.ndim == 1 else samples.mean(axis=1)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    divisor = gcd(source_rate, target_rate)
    return resample_poly(samples, target_rate // divisor, source_rate // divisor)


def _frame_dbfs(samples: np.ndarray, rate: int) -> np.ndarray:
    size = max(1, int(rate * FRAME_SECONDS))
    count = len(samples) // size
    if count == 0:
        return np.array([])
    frames = samples[: count * size].reshape(count, size)
    rms = np.sqrt((frames**2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10))


def trim_silence(samples: np.ndarray, rate: int) -> np.ndarray:
    """Cuts leading and trailing frames quieter than SILENCE_DBFS, keeping 50 ms of padding."""
    levels = _frame_dbfs(samples, rate)
    loud = np.flatnonzero(levels > SILENCE_DBFS)
    if loud.size == 0:
        return samples[:0]
    size = int(rate * FRAME_SECONDS)
    pad = int(rate * 0.05)
    start = max(0, loud[0] * size - pad)
    end = min(len(samples), (loud[-1] + 1) * size + pad)
    return samples[start:end]


def normalise_loudness(samples: np.ndarray) -> np.ndarray:
    """Scales to TARGET_RMS_DBFS, then limits the peak so nothing clips."""
    rms = float(np.sqrt((samples**2).mean()))
    if rms < 1e-6:
        return samples
    scaled = samples * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = float(np.abs(scaled).max())
    return scaled / peak * 0.97 if peak > 0.97 else scaled


def duration(samples: np.ndarray, rate: int) -> float:
    return len(samples) / rate
