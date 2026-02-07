from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import istft, stft


def peak_normalize_buffer(buffer: np.ndarray, target_peak: float = 0.99) -> np.ndarray:
    if buffer.size == 0:
        return buffer.astype(np.float32)

    peak = float(np.max(np.abs(buffer)))
    if not np.isfinite(peak) or peak <= 0.0:
        return buffer.astype(np.float32)

    normalized = buffer.astype(np.float32) * (target_peak / peak)
    return np.asarray(np.clip(normalized, -1.0, 1.0), dtype=np.float32)


def time_stretch_pitch_preserving(
    buffer: np.ndarray,
    speed_ratio: float,
    sample_rate: int,
    mode: str = "librosa",
) -> np.ndarray:
    if buffer.size == 0:
        return buffer.astype(np.float32)

    ratio = float(speed_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return buffer.astype(np.float32)
    if abs(ratio - 1.0) <= 1e-4:
        return buffer.astype(np.float32)
    if mode == "native":
        return _time_stretch_native_phase_vocoder(buffer, ratio, sample_rate)
    if mode == "librosa":
        return _time_stretch_librosa(buffer, ratio)
    raise ValueError(f"Unknown time-stretch mode: {mode}")


def _time_stretch_librosa(buffer: np.ndarray, ratio: float) -> np.ndarray:
    mono = np.asarray(buffer, dtype=np.float32)
    stretched_wave = librosa.effects.time_stretch(y=mono, rate=ratio)

    target_len = max(1, int(np.round(mono.size / ratio)))
    if stretched_wave.size < target_len:
        stretched_wave = np.pad(stretched_wave, (0, target_len - stretched_wave.size))
    elif stretched_wave.size > target_len:
        stretched_wave = stretched_wave[:target_len]
    return np.asarray(stretched_wave, dtype=np.float32)


def _time_stretch_native_phase_vocoder(buffer: np.ndarray, ratio: float, sample_rate: int) -> np.ndarray:
    mono = np.asarray(buffer, dtype=np.float32)
    n_fft = 2048 if mono.size >= 2048 else max(256, int(2 ** np.floor(np.log2(max(32, mono.size)))))
    hop = max(1, n_fft // 4)
    window = "hann"

    _, _, spectrum = stft(
        mono,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
    )
    if spectrum.shape[1] < 2:
        return mono.astype(np.float32)

    frame_count = spectrum.shape[1]
    time_steps = np.arange(0.0, frame_count - 1, ratio, dtype=np.float64)
    if time_steps.size == 0:
        time_steps = np.array([0.0], dtype=np.float64)

    phase_advance = (2.0 * np.pi * hop * np.arange(spectrum.shape[0], dtype=np.float64)) / float(n_fft)
    phase_acc = np.angle(spectrum[:, 0]).astype(np.float64)
    stretched = np.zeros((spectrum.shape[0], time_steps.size), dtype=np.complex128)

    for frame_index, step in enumerate(time_steps):
        left = min(int(np.floor(step)), frame_count - 2)
        right = left + 1
        frac = float(step - left)
        left_col = spectrum[:, left]
        right_col = spectrum[:, right]
        magnitude = (1.0 - frac) * np.abs(left_col) + frac * np.abs(right_col)

        delta = np.angle(right_col) - np.angle(left_col) - phase_advance
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase_acc += phase_advance + delta
        stretched[:, frame_index] = magnitude * np.exp(1j * phase_acc)

    _, stretched_wave = istft(
        stretched,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )

    target_len = max(1, int(np.round(mono.size / ratio)))
    if stretched_wave.size < target_len:
        stretched_wave = np.pad(stretched_wave, (0, target_len - stretched_wave.size))
    elif stretched_wave.size > target_len:
        stretched_wave = stretched_wave[:target_len]
    return np.asarray(stretched_wave, dtype=np.float32)
