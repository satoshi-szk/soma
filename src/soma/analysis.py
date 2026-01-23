from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
from scipy.io import wavfile
from scipy.signal import find_peaks, resample, resample_poly

from soma.models import AnalysisSettings, AudioInfo, PartialPoint, SpectrogramPreview


def load_audio(path: Path, max_duration_sec: float | None = None) -> tuple[AudioInfo, np.ndarray]:
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


def make_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
    amp_reference: float | None = None,
) -> tuple[SpectrogramPreview, float]:
    """Generate a CWT spectrogram for the specified time/frequency region.

    The computation cost is proportional to the output size (width x height),
    not the audio duration. This allows efficient rendering of both overview
    and zoomed-in views.
    """
    total_duration = audio.shape[0] / float(sample_rate) if audio.size > 0 else 0.0

    if audio.size == 0 or time_end <= time_start:
        return SpectrogramPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            freq_min=freq_min,
            freq_max=freq_max,
            duration_sec=total_duration,
            time_start=time_start,
            time_end=time_end,
        ), 1.0

    # Clamp time range to valid bounds
    time_start = max(0.0, min(time_start, total_duration))
    time_end = max(time_start + 1e-6, min(time_end, total_duration))

    # Extract audio segment for the requested time range
    start_sample = int(time_start * sample_rate)
    end_sample = int(time_end * sample_rate)
    audio_segment = audio[start_sample:end_sample]

    if audio_segment.size < 32:
        return SpectrogramPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            freq_min=freq_min,
            freq_max=freq_max,
            duration_sec=total_duration,
            time_start=time_start,
            time_end=time_end,
        ), 1.0

    # Clamp frequency range
    effective_freq_min = max(freq_min, settings.freq_min, 1.0)
    effective_freq_max = min(freq_max, settings.freq_max, sample_rate * 0.5)
    effective_freq_max = max(effective_freq_max, effective_freq_min * 1.001)

    # Downsample audio so we keep enough temporal samples per output pixel, while
    # preserving a minimum sample rate for the requested freq_max (Nyquist).
    samples_per_pixel = 32
    segment_duration = time_end - time_start
    min_sample_rate = min(sample_rate, max(2000.0, effective_freq_max * 2.2))
    min_samples = int(np.ceil(min_sample_rate * segment_duration))
    target_samples = max(width * samples_per_pixel, min_samples)

    if audio_segment.size > target_samples:
        # Downsample in time domain to reduce CWT computation
        audio_segment = resample(audio_segment, target_samples).astype(np.float32)
        working_sample_rate = int(target_samples / segment_duration)
    else:
        working_sample_rate = sample_rate

    # Further downsample based on frequency range (Nyquist)
    target_rate = _preview_sample_rate(working_sample_rate, effective_freq_max)
    if target_rate < working_sample_rate:
        audio_segment, _ = resample_audio(audio_segment, working_sample_rate, target_rate)
        working_sample_rate = target_rate

    effective_freq_max = min(effective_freq_max, working_sample_rate * 0.5)

    # Determine bins_per_octave based on output height
    # More pixels = more frequency resolution needed
    octaves = np.log2(effective_freq_max / effective_freq_min)
    bins_per_octave = max(4, min(settings.preview_bins_per_octave, int(height / max(1, octaves))))

    cwt_settings = AnalysisSettings(
        freq_min=effective_freq_min,
        freq_max=effective_freq_max,
        bins_per_octave=bins_per_octave,
    )
    frequencies = _build_frequencies(cwt_settings, max_freq=effective_freq_max)
    cwt_matrix = _cwt_magnitude(
        audio_segment,
        working_sample_rate,
        frequencies,
        wavelet_bandwidth=settings.wavelet_bandwidth,
        wavelet_center_freq=settings.wavelet_center_freq,
    )

    computed_amp_reference = float(np.max(cwt_matrix)) if cwt_matrix.size else 1.0
    if amp_reference is None:
        amp_reference = computed_amp_reference

    magnitude = _normalize_cwt(cwt_matrix, reference_max=amp_reference)
    time_resampled = resample(magnitude, width, axis=1)
    resized = resample(time_resampled, height, axis=0)
    normalized = np.clip(resized, 0.0, 1.0)
    normalized = np.flipud(normalized)

    data = (normalized * 255).astype(np.uint8).flatten().tolist()
    return SpectrogramPreview(
        width=width,
        height=height,
        data=data,
        freq_min=effective_freq_min,
        freq_max=effective_freq_max,
        duration_sec=total_duration,
        time_start=time_start,
        time_end=time_end,
    ), computed_amp_reference


def make_spectrogram_stft(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
    amp_reference: float | None = None,
) -> tuple[SpectrogramPreview, float]:
    """Generate a fast STFT-based preview spectrogram for GUI display only.

    This must never be used for snap/ridge detection.
    """
    total_duration = audio.shape[0] / float(sample_rate) if audio.size > 0 else 0.0

    if audio.size == 0 or time_end <= time_start:
        return SpectrogramPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            freq_min=freq_min,
            freq_max=freq_max,
            duration_sec=total_duration,
            time_start=time_start,
            time_end=time_end,
        ), 1.0

    # Clamp time range to valid bounds
    time_start = max(0.0, min(time_start, total_duration))
    time_end = max(time_start + 1e-6, min(time_end, total_duration))

    start_sample = int(time_start * sample_rate)
    end_sample = int(time_end * sample_rate)
    audio_segment = _to_float32(audio[start_sample:end_sample])

    if audio_segment.size < 32:
        return SpectrogramPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            freq_min=freq_min,
            freq_max=freq_max,
            duration_sec=total_duration,
            time_start=time_start,
            time_end=time_end,
        ), 1.0

    # Clamp frequency range
    effective_freq_min = max(freq_min, settings.freq_min, 1.0)
    effective_freq_max = min(freq_max, settings.preview_freq_max, settings.freq_max, sample_rate * 0.5)
    effective_freq_max = max(effective_freq_max, effective_freq_min * 1.001)

    # Sparse STFT: sample `width` windows across the segment (no sliding),
    # to avoid O(N) work for long audio.
    nperseg = 4096 if sample_rate >= 48000 else 2048
    nperseg = int(min(nperseg, max(256, audio_segment.size)))
    if nperseg & (nperseg - 1) != 0:
        nperseg = 1 << int(np.floor(np.log2(nperseg)))
        nperseg = max(256, nperseg)

    window = np.hanning(nperseg).astype(np.float32)
    fft_freqs = np.fft.rfftfreq(nperseg, d=1.0 / float(sample_rate)).astype(np.float64)
    if fft_freqs.size < 2:
        return SpectrogramPreview(
            width=width,
            height=height,
            data=[0] * (width * height),
            freq_min=effective_freq_min,
            freq_max=effective_freq_max,
            duration_sec=total_duration,
            time_start=time_start,
            time_end=time_end,
        ), 1.0

    frame_centers = np.linspace(0, audio_segment.size - 1, width, dtype=np.int64)
    half = nperseg // 2
    mags = np.zeros((fft_freqs.size, width), dtype=np.float32)

    audio_f64 = audio_segment.astype(np.float64, copy=False)
    for col, center in enumerate(frame_centers.tolist()):
        start = int(center) - half
        end = start + nperseg
        if start < 0 or end > audio_segment.size:
            padded = np.zeros(nperseg, dtype=np.float64)
            src_start = max(0, start)
            src_end = min(audio_segment.size, end)
            dst_start = src_start - start
            dst_end = dst_start + (src_end - src_start)
            padded[dst_start:dst_end] = audio_f64[src_start:src_end]
            segment = padded
        else:
            segment = audio_f64[start:end]
        spectrum = np.fft.rfft(segment * window, n=nperseg)
        mags[:, col] = np.abs(spectrum).astype(np.float32)

    # Interpolate linear-frequency magnitudes to log-frequency axis.
    log_freqs = np.geomspace(effective_freq_min, effective_freq_max, height).astype(np.float64)
    resized = np.empty((height, width), dtype=np.float32)
    for col in range(width):
        resized[:, col] = np.interp(log_freqs, fft_freqs, mags[:, col], left=0.0, right=0.0).astype(np.float32)

    computed_amp_reference = float(np.max(resized)) if resized.size else 1.0
    if amp_reference is None:
        amp_reference = computed_amp_reference

    normalized = _normalize_magnitude_db(resized, reference_max=amp_reference)
    normalized = np.flipud(normalized)
    data = (normalized * 255).astype(np.uint8).flatten().tolist()
    return SpectrogramPreview(
        width=width,
        height=height,
        data=data,
        freq_min=effective_freq_min,
        freq_max=effective_freq_max,
        duration_sec=total_duration,
        time_start=time_start,
        time_end=time_end,
    ), computed_amp_reference


def snap_trace(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    trace: Sequence[tuple[float, float]],
    window_ms: float = 200.0,
    freq_window_octaves: float = 0.5,
    amp_reference: float | None = None,
) -> list[PartialPoint]:
    if not trace or audio.size == 0:
        return []

    resampled = _resample_trace(trace, settings.time_resolution_ms)
    trace_points = list(resampled)

    window_samples = int(sample_rate * window_ms / 1000.0)
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
        frequencies = _build_frequencies(
            AnalysisSettings(
                freq_min=local_freq_min,
                freq_max=local_freq_max,
                bins_per_octave=settings.bins_per_octave,
            )
        )

        magnitude = _cwt_magnitude(
            window,
            sample_rate,
            frequencies,
            wavelet_bandwidth=settings.wavelet_bandwidth,
            wavelet_center_freq=settings.wavelet_center_freq,
        )
        center_idx = min(magnitude.shape[1] - 1, window.shape[0] // 2)
        spectrum = magnitude[:, center_idx]
        if spectrum.size == 0:
            continue

        # Find local maxima (peaks where both neighbors are smaller)
        peak_indices, _ = find_peaks(spectrum)
        if len(peak_indices) > 0:
            # Select the peak closest to freq_in (in log-frequency space)
            peak_freqs = frequencies[peak_indices]
            log_distances = np.abs(np.log2(peak_freqs / freq_in))
            closest_idx = peak_indices[int(np.argmin(log_distances))]
        else:
            # Fallback: no local maxima found, use global max
            closest_idx = int(np.argmax(spectrum))
        peak_freq = float(frequencies[closest_idx])
        peak_amp = float(spectrum[closest_idx])
        window_max = float(np.max(magnitude))
        normalizer = amp_reference if amp_reference and amp_reference > 0 else window_max
        normalized_amp = float(np.clip(peak_amp / (normalizer + 1e-8), 0.0, 1.0))
        points.append(PartialPoint(time=time_sec, freq=peak_freq, amp=normalized_amp))

    return points


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


def _cwt_magnitude(
    audio: np.ndarray,
    sample_rate: float,
    frequencies: np.ndarray,
    wavelet_bandwidth: float = 8.0,
    wavelet_center_freq: float = 1.0,
) -> np.ndarray:
    # Bandwidth parameter controls time/frequency trade-off.
    # A larger value yields sharper frequency ridges (at the cost of time smearing),
    # which better matches typical "thin ridge" spectrogram expectations.
    bw = float(wavelet_bandwidth)
    # Keep it in a sensible range; this is a UI-controlled parameter and should never explode.
    bw = 0.1 if not np.isfinite(bw) or bw <= 0 else min(bw, 64.0)
    cf = float(wavelet_center_freq)
    cf = 0.1 if not np.isfinite(cf) or cf <= 0 else min(cf, 16.0)
    wavelet = pywt.ContinuousWavelet(f"cmor{bw:.1f}-{cf:.1f}")
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq * sample_rate / frequencies
    coefficients, _ = pywt.cwt(audio, scales, wavelet, sampling_period=1.0 / sample_rate)
    coefficients = np.asarray(coefficients)
    return np.asarray(np.abs(coefficients), dtype=np.float32)


def _normalize_cwt(magnitude: np.ndarray, reference_max: float | None = None) -> np.ndarray:
    """Normalize CWT magnitude for visualization.

    We use a fixed dB range relative to the peak instead of min/max normalization.
    Min/max tends to lift the noise floor (especially for narrowband signals like a sine),
    making ridges look unrealistically thick.

    Args:
        magnitude: CWT magnitude matrix.
        reference_max: If provided, use this as the reference maximum instead of
            computing from the magnitude. This ensures consistent brightness
            across different zoom levels.
    """
    if magnitude.size == 0:
        return np.zeros_like(magnitude, dtype=np.float32)

    # Visual dynamic range in dB (0dB = peak, <= min_db -> black).
    min_db = -60.0
    eps = 1e-12
    max_mag = reference_max if reference_max is not None and reference_max > 0 else float(np.max(magnitude))
    if max_mag <= 0:
        return np.zeros_like(magnitude, dtype=np.float32)

    db = 20.0 * (np.log10(magnitude.astype(np.float64) + eps) - np.log10(max_mag + eps))
    normalized = (db - min_db) / (-min_db)
    return np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)


def _normalize_magnitude_db(magnitude: np.ndarray, reference_max: float | None = None) -> np.ndarray:
    if magnitude.size == 0:
        return np.zeros_like(magnitude, dtype=np.float32)

    min_db = -60.0
    eps = 1e-12
    max_mag = reference_max if reference_max is not None and reference_max > 0 else float(np.max(magnitude))
    if max_mag <= 0:
        return np.zeros_like(magnitude, dtype=np.float32)

    db = 20.0 * (np.log10(magnitude.astype(np.float64) + eps) - np.log10(max_mag + eps))
    normalized = (db - min_db) / (-min_db)
    return np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)


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
