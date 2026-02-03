from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import find_peaks, resample, resample_poly

from soma.models import AnalysisSettings, AudioInfo, PartialPoint, SpectrogramPreview


def load_audio(
    path: Path,
    max_duration_sec: float | None = None,
    display_name: str | None = None,
) -> tuple[AudioInfo, np.ndarray]:
    try:
        sample_rate, channels, duration_sec, truncated, audio = _read_audio_soundfile(path, max_duration_sec)
    except Exception:
        if not _should_use_audioread(path):
            raise
        sample_rate, channels, duration_sec, truncated, audio = _read_audio_audioread(path, max_duration_sec)

    info = AudioInfo(
        path=str(path),
        name=display_name or path.name,
        sample_rate=int(sample_rate),
        duration_sec=duration_sec,
        channels=channels,
        truncated=truncated,
    )
    return info, audio


def get_audio_duration_sec(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        frames = int(info.frames)
        samplerate = int(info.samplerate)
        if frames > 0 and samplerate > 0:
            return frames / float(samplerate)
    except Exception:
        pass

    if not _should_use_audioread(path):
        raise ValueError("Failed to read audio header")

    try:
        import audioread

        with audioread.audio_open(str(path)) as handle:
            if handle.duration and handle.duration > 0:
                return float(handle.duration)
    except Exception as exc:
        raise ValueError("Failed to read audio header") from exc

    raise ValueError("Failed to read audio header")


def _read_audio_soundfile(
    path: Path,
    max_duration_sec: float | None,
) -> tuple[int, int, float, bool, np.ndarray]:
    import soundfile as sf

    info = sf.info(path)
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    total_samples = info.frames if info.frames > 0 else None
    max_samples = int(sample_rate * max_duration_sec) if max_duration_sec is not None else None
    truncated = False
    frames_to_read = -1

    if max_samples is not None and (total_samples is None or total_samples > max_samples):
        frames_to_read = max_samples
        truncated = True

    with sf.SoundFile(path) as handle:
        audio = handle.read(frames=frames_to_read, dtype="float32", always_2d=True)

    mono = np.zeros(0, dtype=np.float32) if audio.size == 0 else audio.mean(axis=1).astype(np.float32)

    if total_samples is None:
        total_samples = mono.shape[0]
        if max_samples is not None and mono.shape[0] >= max_samples:
            truncated = True

    duration_sec = total_samples / float(sample_rate) if sample_rate > 0 else 0.0
    return sample_rate, channels, duration_sec, truncated, mono


def _read_audio_audioread(
    path: Path,
    max_duration_sec: float | None,
) -> tuple[int, int, float, bool, np.ndarray]:
    import audioread

    with audioread.audio_open(str(path)) as handle:
        sample_rate = int(handle.samplerate)
        channels = int(handle.channels)
        duration_sec = float(handle.duration) if handle.duration else 0.0
        max_samples = int(sample_rate * max_duration_sec) if max_duration_sec is not None else None
        total_samples = int(duration_sec * sample_rate) if duration_sec > 0 else None
        truncated = False

        chunks: list[np.ndarray] = []
        read_samples = 0
        leftover = b""
        bytes_per_sample = 2
        frame_bytes = max(1, channels * bytes_per_sample)
        for buffer in handle:
            data = leftover + buffer
            frames = len(data) // frame_bytes
            if frames == 0:
                leftover = data
                continue
            data_used = data[: frames * frame_bytes]
            leftover = data[frames * frame_bytes :]

            chunk = np.frombuffer(data_used, dtype="<i2")
            if channels > 1:
                chunk = chunk.reshape(frames, channels).mean(axis=1)
            if max_samples is not None:
                remaining = max_samples - read_samples
                if remaining <= 0:
                    truncated = True
                    break
                if chunk.shape[0] > remaining:
                    chunk = chunk[:remaining]
                    truncated = True
            chunks.append(chunk.astype(np.float32))
            read_samples += chunk.shape[0]

        if chunks:
            audio = np.concatenate(chunks)
            audio = (audio / 32768.0).clip(-1.0, 1.0).astype(np.float32)
        else:
            audio = np.zeros(0, dtype=np.float32)

        if total_samples is None:
            total_samples = read_samples
        if max_samples is not None and total_samples > max_samples:
            truncated = True

        duration_sec = total_samples / float(sample_rate) if sample_rate > 0 else 0.0
        return sample_rate, channels, duration_sec, truncated, audio


def _should_use_audioread(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in {
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
        ".mp4",
        ".m4b",
        ".m4p",
    }


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

    # 時間範囲を有効な境界内に収める。
    time_start = max(0.0, min(time_start, total_duration))
    time_end = max(time_start + 1e-6, min(time_end, total_duration))

    # 要求された時間範囲の音声セグメントを切り出す。
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

    # 周波数範囲を有効な境界内に収める。
    effective_freq_min = max(freq_min, settings.freq_min, 1.0)
    effective_freq_max = min(freq_max, settings.freq_max, sample_rate * 0.5)
    effective_freq_max = max(effective_freq_max, effective_freq_min * 1.001)

    # 出力 1px あたりの時間サンプル数を確保しつつ、要求 freq_max の
    # ナイキスト条件を満たす最小サンプルレートを維持するためにダウンサンプルする。
    samples_per_pixel = 32
    segment_duration = time_end - time_start
    min_sample_rate = min(sample_rate, max(2000.0, effective_freq_max * 2.2))
    min_samples = int(np.ceil(min_sample_rate * segment_duration))
    target_samples = max(width * samples_per_pixel, min_samples)

    if audio_segment.size > target_samples:
        # CWT 計算量を減らすため、時間領域でダウンサンプルする。
        audio_segment = resample(audio_segment, target_samples).astype(np.float32)
        working_sample_rate = int(target_samples / segment_duration)
    else:
        working_sample_rate = sample_rate

    # 周波数範囲（ナイキスト条件）に基づいてさらにダウンサンプルする。
    target_rate = _preview_sample_rate(working_sample_rate, effective_freq_max)
    if target_rate < working_sample_rate:
        audio_segment, _ = resample_audio(audio_segment, working_sample_rate, target_rate)
        working_sample_rate = target_rate

    effective_freq_max = min(effective_freq_max, working_sample_rate * 0.5)

    # 出力高さに応じて bins_per_octave を決める。
    # ピクセル数が多いほど、より高い周波数解像度が必要になる。
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

    # 時間範囲を有効な境界内に収める。
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

    # 周波数範囲を有効な境界内に収める。
    effective_freq_min = max(freq_min, settings.freq_min, 1.0)
    effective_freq_max = min(freq_max, settings.preview_freq_max, settings.freq_max, sample_rate * 0.5)
    effective_freq_max = max(effective_freq_max, effective_freq_min * 1.001)

    # 疎な STFT: セグメント全体から `width` 個の窓を抽出する（スライドしない）。
    # 長い音声で O(N) の処理量になることを避けるため。
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

    # 線形周波数軸の振幅を対数周波数軸へ補間する。
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


def _stft_peak_location(
    audio_segment: np.ndarray,
    sample_rate: int,
    freq_min: float,
    freq_max: float,
    width: int,
) -> tuple[int, float]:
    if audio_segment.size < 32:
        return 0, float(freq_min)

    nperseg = 4096 if sample_rate >= 48000 else 2048
    nperseg = int(min(nperseg, max(256, audio_segment.size)))
    if nperseg & (nperseg - 1) != 0:
        nperseg = 1 << int(np.floor(np.log2(nperseg)))
        nperseg = max(256, nperseg)

    window = np.hanning(nperseg).astype(np.float64)
    fft_freqs = np.fft.rfftfreq(nperseg, d=1.0 / float(sample_rate)).astype(np.float64)
    if fft_freqs.size < 2:
        return 0, float(freq_min)

    freq_mask = (fft_freqs >= freq_min) & (fft_freqs <= freq_max)
    if not np.any(freq_mask):
        freq_mask = np.ones_like(fft_freqs, dtype=bool)

    frame_centers = np.linspace(0, audio_segment.size - 1, max(1, width), dtype=np.int64)
    half = nperseg // 2
    best_mag = -1.0
    best_center = int(frame_centers[0]) if frame_centers.size else 0
    best_freq = float(freq_min)

    audio_f64 = audio_segment.astype(np.float64, copy=False)
    freq_indices = np.flatnonzero(freq_mask)
    for center in frame_centers.tolist():
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
        spectrum = np.abs(np.fft.rfft(segment * window, n=nperseg))
        masked = spectrum[freq_indices]
        if masked.size == 0:
            continue
        local_idx = int(np.argmax(masked))
        local_mag = float(masked[local_idx])
        if local_mag > best_mag:
            best_mag = local_mag
            best_center = int(center)
            best_freq = float(fft_freqs[freq_indices[local_idx]])

    return best_center, best_freq


def estimate_cwt_amp_reference(
    audio: np.ndarray,
    sample_rate: int,
    settings: AnalysisSettings,
    time_start: float = 0.0,
    time_end: float | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    stft_width: int = 512,
    window_ms: float = 200.0,
) -> float:
    """Estimate a stable CWT amplitude reference using STFT for localization."""
    if audio.size == 0:
        return 1.0

    total_duration = audio.shape[0] / float(sample_rate)
    start_time = max(0.0, min(time_start, total_duration))
    end_time = total_duration if time_end is None else max(start_time + 1e-6, min(time_end, total_duration))
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    audio_segment = _to_float32(audio[start_sample:end_sample])
    if audio_segment.size < 32:
        return 1.0

    effective_freq_min = max(settings.freq_min, 1.0) if freq_min is None else max(freq_min, 1.0)
    effective_freq_max = (
        min(settings.freq_max, sample_rate * 0.5) if freq_max is None else min(freq_max, sample_rate * 0.5)
    )
    effective_freq_max = max(effective_freq_max, effective_freq_min * 1.001)

    peak_center, _ = _stft_peak_location(
        audio_segment,
        sample_rate,
        effective_freq_min,
        effective_freq_max,
        width=stft_width,
    )
    window_samples = max(32, int(sample_rate * window_ms / 1000.0))
    half_window = window_samples // 2
    center_sample = start_sample + peak_center
    window_start = max(0, center_sample - half_window)
    window_end = min(audio.shape[0], center_sample + half_window)
    window = _to_float32(audio[window_start:window_end])
    if window.size < 32:
        return 1.0

    frequencies = _build_frequencies(
        AnalysisSettings(
            freq_min=effective_freq_min,
            freq_max=effective_freq_max,
            bins_per_octave=settings.bins_per_octave,
            wavelet_bandwidth=settings.wavelet_bandwidth,
            wavelet_center_freq=settings.wavelet_center_freq,
        ),
        max_freq=effective_freq_max,
    )
    magnitude = _cwt_magnitude(
        window,
        sample_rate,
        frequencies,
        wavelet_bandwidth=settings.wavelet_bandwidth,
        wavelet_center_freq=settings.wavelet_center_freq,
    )
    max_mag = float(np.max(magnitude)) if magnitude.size else 0.0
    return max_mag if max_mag > 0 else 1.0


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

        # 局所最大を探す（左右の近傍より大きいピーク）。
        peak_indices, _ = find_peaks(spectrum)
        if len(peak_indices) > 0:
            # 対数周波数空間で freq_in に最も近いピークを選ぶ。
            peak_freqs = frequencies[peak_indices]
            log_distances = np.abs(np.log2(peak_freqs / freq_in))
            closest_idx = peak_indices[int(np.argmin(log_distances))]
        else:
            # フォールバック: 局所最大がなければ全体最大を使う。
            closest_idx = int(np.argmax(spectrum))
        peak_freq = float(frequencies[closest_idx])
        peak_amp = float(spectrum[closest_idx])
        window_max = float(np.max(magnitude))
        normalizer = amp_reference if amp_reference and amp_reference > 0 else window_max
        normalizer = max(normalizer, window_max)
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
    # 帯域幅パラメータは時間/周波数のトレードオフを制御する。
    # 値を大きくすると周波数方向のリッジが鋭くなる一方、時間方向はにじみやすくなる。
    # 典型的な「細いリッジ」のスペクトログラム表現に合わせるための調整。
    bw = float(wavelet_bandwidth)
    # UI から変更可能な値なので、暴走しないよう妥当な範囲に制限する。
    bw = 0.1 if not np.isfinite(bw) or bw <= 0 else min(bw, 64.0)
    cf = float(wavelet_center_freq)
    cf = 0.1 if not np.isfinite(cf) or cf <= 0 else min(cf, 16.0)
    wavelet = pywt.ContinuousWavelet(f"cmor{bw:.1f}-{cf:.1f}")
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq * sample_rate / frequencies
    coefficients, _ = pywt.cwt(
        audio,
        scales,
        wavelet,
        sampling_period=1.0 / sample_rate,
        method="fft",
    )
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

    # 表示用ダイナミックレンジ（dB）。0dB はピーク、min_db 以下は黒。
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
