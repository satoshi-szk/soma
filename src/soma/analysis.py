from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
from scipy.io import wavfile
from scipy.signal import resample, resample_poly

from soma.models import AnalysisSettings, AudioInfo, PartialPoint, SpectrogramPreview, ViewportPreview


def load_audio(path: Path, max_duration_sec: float | None = 30.0) -> tuple[AudioInfo, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    if raw.ndim == 1:
        channels = 1
        audio = _to_float32(raw)
    else:
        channels = raw.shape[1]
        audio = _to_float32(raw).mean(axis=1).astype(np.float32)
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
) -> tuple[SpectrogramPreview, float]:
    if audio.size == 0:
        return SpectrogramPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            freq_min=settings.freq_min,
            freq_max=settings.freq_max,
            duration_sec=0.0,
        ), 1.0

    original_duration = audio.shape[0] / float(sample_rate)
    preview_freq_max = min(settings.preview_freq_max, settings.freq_max)
    preview_freq_max = max(preview_freq_max, settings.freq_min)
    preview_bins_per_octave = max(4, int(settings.preview_bins_per_octave))
    target_rate = _preview_sample_rate(sample_rate, preview_freq_max)
    if target_rate != sample_rate:
        audio = resample_audio(audio, sample_rate, target_rate)[0]
        sample_rate = target_rate

    preview_freq_max = min(preview_freq_max, sample_rate * 0.5)
    preview_settings = AnalysisSettings(
        freq_min=settings.freq_min,
        freq_max=preview_freq_max,
        bins_per_octave=preview_bins_per_octave,
    )
    frequencies = _build_frequencies(preview_settings, max_freq=preview_freq_max)
    cwt_matrix = _cwt_magnitude(audio, sample_rate, frequencies)
    amp_reference = float(np.max(cwt_matrix)) if cwt_matrix.size else 1.0

    magnitude = _normalize_cwt(cwt_matrix)
    time_resampled = resample(magnitude, width, axis=1)
    resized = resample(time_resampled, height, axis=0)
    resized = resample(resized, width, axis=1)
    normalized = np.clip(resized, 0.0, 1.0)
    normalized = np.flipud(normalized)

    data = (normalized * 255).astype(np.uint8).flatten().tolist()
    duration_sec = original_duration
    return SpectrogramPreview(
        width=width,
        height=height,
        data=data,
        freq_min=settings.freq_min,
        freq_max=preview_freq_max,
        duration_sec=duration_sec,
    ), amp_reference


def make_viewport_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
) -> ViewportPreview:
    """Generate a high-resolution spectrogram for the specified viewport region."""
    if audio.size == 0 or time_end <= time_start:
        return ViewportPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    duration = audio.shape[0] / float(sample_rate)
    time_start = max(0.0, min(time_start, duration))
    time_end = max(time_start, min(time_end, duration))

    start_sample = int(time_start * sample_rate)
    end_sample = int(time_end * sample_rate)
    audio_segment = audio[start_sample:end_sample]

    if audio_segment.size < 32:
        return ViewportPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    effective_freq_min = max(freq_min, settings.freq_min, 1.0)
    effective_freq_max = min(freq_max, settings.freq_max)
    effective_freq_max = max(effective_freq_max, effective_freq_min * 1.001)

    target_rate = _preview_sample_rate(sample_rate, effective_freq_max)
    if target_rate < sample_rate:
        audio_segment, _ = resample_audio(audio_segment, sample_rate, target_rate)
        sample_rate = target_rate

    effective_freq_max = min(effective_freq_max, sample_rate * 0.5)

    viewport_settings = AnalysisSettings(
        freq_min=effective_freq_min,
        freq_max=effective_freq_max,
        bins_per_octave=settings.bins_per_octave,
    )
    frequencies = _build_frequencies(viewport_settings, max_freq=effective_freq_max)
    cwt_matrix = _cwt_magnitude(audio_segment, sample_rate, frequencies)
    magnitude = _normalize_cwt(cwt_matrix)

    time_resampled = resample(magnitude, width, axis=1)
    resized = resample(time_resampled, height, axis=0)
    resized = resample(resized, width, axis=1)
    normalized = np.clip(resized, 0.0, 1.0)
    normalized = np.flipud(normalized)

    data = (normalized * 255).astype(np.uint8).flatten().tolist()
    return ViewportPreview(
        width=width,
        height=height,
        data=data,
        time_start=time_start,
        time_end=time_end,
        freq_min=effective_freq_min,
        freq_max=effective_freq_max,
    )


def snap_trace(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    trace: Sequence[tuple[float, float]],
    window_ms: float = 200.0,
    freq_window_octaves: float = 0.5,
    max_points: int | None = None,
    amp_reference: float | None = None,
) -> list[PartialPoint]:
    if not trace or audio.size == 0:
        return []

    resampled = _resample_trace(trace, settings.time_resolution_ms)
    trace_points = list(resampled)
    use_fast_mode = len(trace_points) > 256
    if max_points is not None and len(trace_points) > max_points:
        trace_points = _decimate_trace(trace_points, max_points=max_points)
        use_fast_mode = True
    effective_bins = settings.bins_per_octave
    effective_window_ms = window_ms
    if use_fast_mode:
        effective_bins = max(8, settings.bins_per_octave // 3)
        effective_window_ms = min(window_ms, 80.0)

    window_samples = int(sample_rate * effective_window_ms / 1000.0)
    window_samples = max(32, window_samples)
    half_window = window_samples // 2

    points: list[PartialPoint] = []
    for time_sec, freq_in in trace_points:
        center = int(time_sec * sample_rate)
        start = max(0, center - half_window)
        end = min(audio.shape[0], center + half_window)
        window = audio[start:end]
        if window.size < 32:
            continue

        window = _to_float32(window)
        local_freq_min = max(1.0, freq_in / (2.0**freq_window_octaves))
        local_freq_max = min(settings.freq_max, freq_in * (2.0**freq_window_octaves))
        window_sample_rate = sample_rate
        target_rate = _preview_sample_rate(window_sample_rate, local_freq_max)
        if target_rate < window_sample_rate:
            window, _ = resample_audio(window, window_sample_rate, target_rate)
            window_sample_rate = target_rate
        frequencies = _build_frequencies(
            AnalysisSettings(
                freq_min=local_freq_min,
                freq_max=local_freq_max,
                bins_per_octave=effective_bins,
            )
        )

        magnitude = _cwt_magnitude(window, window_sample_rate, frequencies)
        center_idx = min(magnitude.shape[1] - 1, window.shape[0] // 2)
        spectrum = magnitude[:, center_idx]
        if spectrum.size == 0:
            continue

        peak_index = int(np.argmax(spectrum))
        peak_freq = float(frequencies[peak_index])
        peak_amp = float(spectrum[peak_index])
        window_max = float(np.max(magnitude))
        normalizer = amp_reference if amp_reference and amp_reference > 0 else window_max
        normalized_amp = float(np.clip(peak_amp / (normalizer + 1e-8), 0.0, 1.0))
        points.append(PartialPoint(time=time_sec, freq=peak_freq, amp=normalized_amp))

    return points


def _decimate_trace(trace: Sequence[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if max_points <= 0:
        return []
    if len(trace) <= max_points:
        return list(trace)
    indices = np.linspace(0, len(trace) - 1, max_points, dtype=int)
    return [trace[idx] for idx in indices]


def _resample_trace(trace: Sequence[tuple[float, float]], time_resolution_ms: float) -> list[tuple[float, float]]:
    if len(trace) < 2 or time_resolution_ms <= 0:
        return list(trace)
    sorted_trace = sorted(trace, key=lambda item: item[0])
    times = np.array([item[0] for item in sorted_trace], dtype=np.float64)
    freqs = np.array([item[1] for item in sorted_trace], dtype=np.float64)
    if np.allclose(times[0], times[-1]):
        return list(trace)
    step = time_resolution_ms / 1000.0
    resampled_times = np.arange(times[0], times[-1] + step * 0.5, step, dtype=np.float64)
    resampled_freqs = np.interp(resampled_times, times, freqs)
    return list(zip(resampled_times.tolist(), resampled_freqs.tolist(), strict=True))


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
