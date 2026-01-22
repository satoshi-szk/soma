from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
from scipy.io import wavfile
from scipy.signal import resample, resample_poly

from soma.models import AnalysisSettings, AudioInfo, PartialPoint, SpectrogramPreview


def load_audio(path: Path, max_duration_sec: float | None = 30.0) -> tuple[AudioInfo, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    if raw.ndim == 1:
        channels = 1
        audio = raw
    else:
        channels = raw.shape[1]
        audio = raw.mean(axis=1)

    audio = _to_float32(audio)
    total_samples = audio.shape[0]
    truncated = False
    if max_duration_sec is not None:
        max_samples = int(sample_rate * max_duration_sec)
        truncated = total_samples > max_samples
        if truncated:
            audio = audio[:max_samples]

    duration_sec = total_samples / float(sample_rate)
    info = AudioInfo(
        path=str(path),
        name=path.name,
        sample_rate=int(sample_rate),
        duration_sec=duration_sec,
        channels=channels,
        truncated=truncated,
    )
    return info, audio


def make_spectrogram_preview(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    width: int = 768,
    height: int = 320,
) -> SpectrogramPreview:
    if audio.size == 0:
        return SpectrogramPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            freq_min=settings.freq_min,
            freq_max=settings.freq_max,
            duration_sec=0.0,
        )

    original_duration = audio.shape[0] / float(sample_rate)
    time_resolution = max(1.0, settings.time_resolution_ms) / 1000.0
    target_bins = max(width, int(np.ceil(original_duration / time_resolution)))
    max_bins = width * 4
    target_bins = min(target_bins, max_bins)

    if audio.shape[0] != target_bins:
        audio = resample(audio, target_bins)
        sample_rate = int(np.ceil(target_bins / original_duration))

    target_rate = _preview_sample_rate(sample_rate, settings.freq_max)
    if target_rate != sample_rate:
        audio = resample_audio(audio, sample_rate, target_rate)[0]
        sample_rate = target_rate

    preview_freq_max = min(settings.freq_max, sample_rate * 0.5)
    frequencies = _build_frequencies(settings, max_freq=preview_freq_max)
    cwt_matrix = _cwt_magnitude(audio, sample_rate, frequencies)

    magnitude = _normalize_cwt(cwt_matrix)
    time_resampled = resample(magnitude, width, axis=1)
    resized = resample(time_resampled, height, axis=0)
    resized = resample(resized, width, axis=1)
    normalized = np.clip(resized, 0.0, 1.0)

    data = (normalized * 255).astype(np.uint8).flatten().tolist()
    duration_sec = original_duration
    return SpectrogramPreview(
        width=width,
        height=height,
        data=data,
        freq_min=settings.freq_min,
        freq_max=preview_freq_max,
        duration_sec=duration_sec,
    )


def snap_trace(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    trace: Sequence[tuple[float, float]],
    window_ms: float = 200.0,
    freq_window_octaves: float = 0.5,
) -> list[PartialPoint]:
    if not trace or audio.size == 0:
        return []

    window_samples = int(sample_rate * window_ms / 1000.0)
    window_samples = max(32, window_samples)
    half_window = window_samples // 2

    points: list[PartialPoint] = []
    for time_sec, freq_in in trace:
        center = int(time_sec * sample_rate)
        start = max(0, center - half_window)
        end = min(audio.shape[0], center + half_window)
        window = audio[start:end]
        if window.size < 32:
            continue

        window = _to_float32(window)
        local_freq_min = max(1.0, freq_in / (2.0**freq_window_octaves))
        local_freq_max = min(settings.freq_max, freq_in * (2.0**freq_window_octaves))
        frequencies = _build_frequencies(
            AnalysisSettings(
                freq_min=local_freq_min,
                freq_max=local_freq_max,
                bins_per_octave=settings.bins_per_octave,
            )
        )

        magnitude = _cwt_magnitude(window, sample_rate, frequencies)
        center_idx = min(magnitude.shape[1] - 1, window.shape[0] // 2)
        spectrum = magnitude[:, center_idx]
        if spectrum.size == 0:
            continue

        peak_index = int(np.argmax(spectrum))
        peak_freq = float(frequencies[peak_index])
        peak_amp = float(spectrum[peak_index])
        normalized_amp = float(peak_amp / (np.max(spectrum) + 1e-8))
        points.append(PartialPoint(time=time_sec, freq=peak_freq, amp=normalized_amp))

    return points


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        return (audio.astype(np.float32) / float(info.max)).clip(-1.0, 1.0)

    return audio.astype(np.float32)


def _build_frequencies(settings: AnalysisSettings, max_freq: float | None = None) -> np.ndarray:
    min_freq = max(1e-3, settings.freq_min)
    max_freq = max(min_freq * 1.001, settings.freq_max if max_freq is None else min(settings.freq_max, max_freq))
    octaves = np.log2(max_freq / min_freq)
    bins = max(8, int(np.ceil(octaves * settings.bins_per_octave)))
    indices = np.arange(bins)
    return min_freq * (2.0 ** (indices / settings.bins_per_octave))


def _cwt_magnitude(audio: np.ndarray, sample_rate: float, frequencies: np.ndarray) -> np.ndarray:
    wavelet = pywt.ContinuousWavelet("cmor1.5-1.0")
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq * sample_rate / frequencies
    coefficients, _ = pywt.cwt(audio, scales, wavelet, sampling_period=1.0 / sample_rate)
    coefficients = np.asarray(coefficients)
    return np.asarray(np.abs(coefficients), dtype=np.float32)


def _normalize_cwt(magnitude: np.ndarray) -> np.ndarray:
    log_mag = np.log10(magnitude + 1e-6)
    log_mag -= float(log_mag.min())
    log_mag /= float(log_mag.max() + 1e-8)
    return log_mag


def _preview_sample_rate(sample_rate: int, freq_max: float) -> int:
    nyquist_target = max(2000.0, freq_max * 2.2)
    return min(sample_rate, int(np.ceil(nyquist_target)))


@dataclass(frozen=True)
class ResamplePlan:
    target_rate: int
    ratio: int
    stride: int


def resample_audio(audio: np.ndarray, sample_rate: int, target_rate: int) -> tuple[np.ndarray, ResamplePlan]:
    if target_rate == sample_rate:
        return audio, ResamplePlan(target_rate=target_rate, ratio=1, stride=1)

    gcd = int(np.gcd(sample_rate, target_rate))
    up = target_rate // gcd
    down = sample_rate // gcd
    resampled = resample_poly(audio, up, down).astype(np.float32)
    plan = ResamplePlan(target_rate=target_rate, ratio=up, stride=down)
    return resampled, plan
