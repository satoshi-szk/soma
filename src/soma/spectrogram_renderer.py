from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import cast

import numpy as np

from soma.analysis import _apply_preview_tone, _normalize_magnitude_db
from soma.models import AnalysisSettings, SpectrogramPreview


class SpectrogramRenderer:
    """Memmap-backed spectrogram renderer."""

    _BASE_HEIGHT = 1024
    _TARGET_COLS_PER_SEC = 84.0
    _LOCAL_ENTER_MULTIPLIER = 1.10
    _LOCAL_EXIT_MULTIPLIER = 0.90
    _LOCAL_CACHE_MAX_ITEMS = 24
    _LOCAL_TIME_QUANTUM_SEC = 0.001
    _LOCAL_FREQ_QUANTUM_HZ = 0.1
    _MIN_COLS = 2048
    _MAX_COLS = 320_000

    def __init__(self, audio: np.ndarray, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._length = int(audio.shape[0])
        self._temp_dir = tempfile.TemporaryDirectory(prefix="soma-spec-")
        self._audio_path = Path(self._temp_dir.name) / "audio.dat"
        self._matrix_path = Path(self._temp_dir.name) / "spec.dat"
        self._closed = False
        try:
            write_map = np.memmap(self._audio_path, dtype=np.float32, mode="w+", shape=(self._length,))
            write_map[:] = np.asarray(audio, dtype=np.float32)
            write_map.flush()
            del write_map
            self._audio = np.memmap(self._audio_path, dtype=np.float32, mode="r", shape=(self._length,))
            self._prefer_local_mode = False
            self._local_cache: OrderedDict[
                tuple[int, int, int, int, int, int, int, int, int, int],
                tuple[SpectrogramPreview, float],
            ] = OrderedDict()
            self._prepare_global_matrix()
        except Exception:
            self._temp_dir.cleanup()
            raise

    def close(self) -> None:
        if self._closed:
            return
        if hasattr(self, "_audio"):
            del self._audio
        if hasattr(self, "_matrix"):
            del self._matrix
        self._temp_dir.cleanup()
        self._closed = True

    def render_overview(
        self,
        settings: AnalysisSettings,
        width: int,
        height: int,
        stft_amp_reference: float | None = None,
    ) -> tuple[SpectrogramPreview, float]:
        duration = self._length / float(self._sample_rate) if self._sample_rate > 0 else 0.0
        return self._render_from_global_matrix(
            settings=settings,
            time_start=0.0,
            time_end=duration,
            freq_min=settings.freq_min,
            freq_max=min(settings.preview_freq_max, settings.freq_max),
            width=width,
            height=height,
            amp_reference=stft_amp_reference,
        )

    def render_tile(
        self,
        settings: AnalysisSettings,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
        stft_amp_reference: float | None = None,
        cwt_amp_reference: float | None = None,
    ) -> tuple[SpectrogramPreview, float, str]:
        time_span = max(time_end - time_start, 1e-9)
        required_cols_per_sec = width / time_span
        use_local = self._should_use_local_mode(required_cols_per_sec)
        amp_reference = stft_amp_reference if stft_amp_reference is not None else cwt_amp_reference

        if use_local:
            preview, ref = self._render_local_tile(
                settings=settings,
                time_start=time_start,
                time_end=time_end,
                freq_min=freq_min,
                freq_max=freq_max,
                width=width,
                height=height,
                amp_reference=amp_reference,
            )
            return preview, ref, "local"

        preview, ref = self._render_from_global_matrix(
            settings=settings,
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
            width=width,
            height=height,
            amp_reference=amp_reference,
        )
        return preview, ref, "high"

    def _should_use_local_mode(self, required_cols_per_sec: float) -> bool:
        enter_threshold = self._TARGET_COLS_PER_SEC * self._LOCAL_ENTER_MULTIPLIER
        exit_threshold = self._TARGET_COLS_PER_SEC * self._LOCAL_EXIT_MULTIPLIER
        if self._prefer_local_mode:
            if required_cols_per_sec < exit_threshold:
                self._prefer_local_mode = False
        elif required_cols_per_sec > enter_threshold:
            self._prefer_local_mode = True
        return self._prefer_local_mode

    def _render_local_tile(
        self,
        settings: AnalysisSettings,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
        amp_reference: float | None,
    ) -> tuple[SpectrogramPreview, float]:
        cache_key = self._build_local_cache_key(
            settings=settings,
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
            width=width,
            height=height,
        )
        cached = self._local_cache.get(cache_key)
        if cached is not None:
            self._local_cache.move_to_end(cache_key)
            return cached

        total_duration = self._length / float(self._sample_rate) if self._sample_rate > 0 else 0.0
        if self._length == 0 or self._sample_rate <= 0 or time_end <= time_start:
            preview = SpectrogramPreview(
                width=width,
                height=height,
                data=[0] * (width * height),
                freq_min=freq_min,
                freq_max=freq_max,
                duration_sec=total_duration,
                time_start=time_start,
                time_end=time_end,
            )
            result = (preview, 1.0)
            self._store_local_cache(cache_key, result)
            return result

        start = max(0.0, min(time_start, total_duration))
        end = max(start + 1e-6, min(time_end, total_duration))
        nyquist = float(self._sample_rate) * 0.5
        effective_min = max(freq_min, settings.freq_min, 20.0)
        effective_max = min(freq_max, settings.preview_freq_max, settings.freq_max, nyquist)
        effective_max = max(effective_max, effective_min * 1.001)

        max_nperseg = 4096
        margin_sec = max((max_nperseg / float(self._sample_rate)) * 0.5, (end - start) * 0.05, 0.01)
        ext_start = max(0.0, start - margin_sec)
        ext_end = min(total_duration, end + margin_sec)
        ext_start_idx = int(np.floor(ext_start * self._sample_rate))
        ext_end_idx = int(np.ceil(ext_end * self._sample_rate))
        ext_end_idx = max(ext_end_idx, ext_start_idx + 1)
        local_audio = np.asarray(self._audio[ext_start_idx:ext_end_idx], dtype=np.float32)
        if local_audio.size == 0:
            preview = SpectrogramPreview(
                width=width,
                height=height,
                data=[0] * (width * height),
                freq_min=effective_min,
                freq_max=effective_max,
                duration_sec=total_duration,
                time_start=start,
                time_end=end,
            )
            result = (preview, 1.0)
            self._store_local_cache(cache_key, result)
            return result

        if width <= 1:
            center_times = np.array([0.5 * (start + end)], dtype=np.float64)
        else:
            center_times = np.linspace(start, end, width, dtype=np.float64)
        centers = np.rint(center_times * float(self._sample_rate)).astype(np.int64) - np.int64(ext_start_idx)
        target_log_freqs = np.geomspace(effective_min, effective_max, height).astype(np.float64)
        band_freq_max = max(effective_max, nyquist)
        weights = self._band_weights(target_log_freqs, band_freq_max)
        composed = np.zeros((height, width), dtype=np.float32)

        bands = [
            (20.0, 200.0, 4096),
            (200.0, 2000.0, 1024),
            (2000.0, band_freq_max, 256),
        ]
        for band_index, (_band_lo, _band_hi, nperseg) in enumerate(bands):
            band_mag = self._sparse_stft_magnitude(local_audio, centers=centers, nperseg=nperseg)
            if band_mag.size == 0:
                continue
            fft_freqs = np.fft.rfftfreq(
                band_mag.shape[0] * 2 - 2,
                d=1.0 / float(self._sample_rate),
            ).astype(np.float64)
            band_image = self._interpolate_freq_matrix(
                source_freqs=fft_freqs,
                source_matrix=band_mag,
                target_freqs=target_log_freqs,
            )
            composed += band_image * weights[band_index][:, None]

        computed_amp_reference = self._matrix_max
        reference = computed_amp_reference if amp_reference is None else amp_reference
        normalized = _normalize_magnitude_db(composed, reference_max=reference)
        normalized = np.flipud(normalized)
        normalized = _apply_preview_tone(normalized, settings)
        data = (normalized * 255).astype(np.uint8).flatten().tolist()
        preview = SpectrogramPreview(
            width=width,
            height=height,
            data=data,
            freq_min=effective_min,
            freq_max=effective_max,
            duration_sec=total_duration,
            time_start=start,
            time_end=end,
        )
        result = (preview, computed_amp_reference)
        self._store_local_cache(cache_key, result)
        return result

    def _build_local_cache_key(
        self,
        settings: AnalysisSettings,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        return (
            int(round(time_start / self._LOCAL_TIME_QUANTUM_SEC)),
            int(round(time_end / self._LOCAL_TIME_QUANTUM_SEC)),
            int(round(freq_min / self._LOCAL_FREQ_QUANTUM_HZ)),
            int(round(freq_max / self._LOCAL_FREQ_QUANTUM_HZ)),
            int(width),
            int(height),
            int(round(settings.gain * 1000.0)),
            int(round(settings.min_db * 10.0)),
            int(round(settings.max_db * 10.0)),
            int(round(settings.gamma * 1000.0)),
        )

    def _store_local_cache(
        self,
        cache_key: tuple[int, int, int, int, int, int, int, int, int, int],
        value: tuple[SpectrogramPreview, float],
    ) -> None:
        self._local_cache[cache_key] = value
        self._local_cache.move_to_end(cache_key)
        while len(self._local_cache) > self._LOCAL_CACHE_MAX_ITEMS:
            self._local_cache.popitem(last=False)

    def _prepare_global_matrix(self) -> None:
        if self._length == 0 or self._sample_rate <= 0:
            self._total_duration = 0.0
            self._matrix_cols = self._MIN_COLS
            self._matrix_freq_min = 20.0
            self._matrix_freq_max = 20000.0
            write_map = np.memmap(
                self._matrix_path,
                dtype=np.float32,
                mode="w+",
                shape=(self._BASE_HEIGHT, self._matrix_cols),
            )
            write_map[:] = 0.0
            write_map.flush()
            del write_map
            self._matrix = np.memmap(
                self._matrix_path,
                dtype=np.float32,
                mode="r",
                shape=(self._BASE_HEIGHT, self._matrix_cols),
            )
            self._matrix_log_freqs = np.geomspace(
                self._matrix_freq_min,
                self._matrix_freq_max,
                self._BASE_HEIGHT,
            ).astype(np.float64)
            self._matrix_max = 1.0
            return

        self._total_duration = self._length / float(self._sample_rate)
        nyquist = float(self._sample_rate) * 0.5
        self._matrix_freq_min = 20.0
        self._matrix_freq_max = max(self._matrix_freq_min * 1.001, nyquist)
        self._matrix_log_freqs = np.geomspace(
            self._matrix_freq_min,
            self._matrix_freq_max,
            self._BASE_HEIGHT,
        ).astype(np.float64)
        estimated_cols = int(round(self._total_duration * self._TARGET_COLS_PER_SEC))
        self._matrix_cols = int(np.clip(estimated_cols, self._MIN_COLS, self._MAX_COLS))
        write_map = np.memmap(
            self._matrix_path,
            dtype=np.float32,
            mode="w+",
            shape=(self._BASE_HEIGHT, self._matrix_cols),
        )
        write_map[:] = 0.0
        centers = np.linspace(0, max(0, self._length - 1), self._matrix_cols, dtype=np.int64)

        bands = [
            (20.0, 200.0, 4096),
            (200.0, 2000.0, 1024),
            (2000.0, self._matrix_freq_max, 256),
        ]

        weights = self._band_weights(self._matrix_log_freqs, self._matrix_freq_max)

        chunk_cols = 256
        for chunk_start in range(0, self._matrix_cols, chunk_cols):
            chunk_end = min(self._matrix_cols, chunk_start + chunk_cols)
            chunk_centers = centers[chunk_start:chunk_end]
            composed_chunk = np.zeros((self._BASE_HEIGHT, chunk_end - chunk_start), dtype=np.float32)
            for band_index, (_band_lo, _band_hi, nperseg) in enumerate(bands):
                band_mag = self._sparse_stft_magnitude(self._audio, centers=chunk_centers, nperseg=nperseg)
                if band_mag.size == 0:
                    continue
                fft_freqs = np.fft.rfftfreq(
                    band_mag.shape[0] * 2 - 2,
                    d=1.0 / float(self._sample_rate),
                ).astype(np.float64)
                band_image = self._interpolate_freq_matrix(
                    source_freqs=fft_freqs,
                    source_matrix=band_mag,
                    target_freqs=self._matrix_log_freqs,
                )
                composed_chunk += band_image * weights[band_index][:, None]
            write_map[:, chunk_start:chunk_end] = composed_chunk
        write_map.flush()
        self._matrix_max = float(np.max(write_map)) if write_map.size else 1.0
        if self._matrix_max <= 0.0:
            self._matrix_max = 1.0
        del write_map
        self._matrix = np.memmap(
            self._matrix_path,
            dtype=np.float32,
            mode="r",
            shape=(self._BASE_HEIGHT, self._matrix_cols),
        )

    def _render_from_global_matrix(
        self,
        settings: AnalysisSettings,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
        amp_reference: float | None = None,
    ) -> tuple[SpectrogramPreview, float]:
        total_duration = self._total_duration
        if self._length == 0 or self._matrix_cols <= 0 or time_end <= time_start:
            return (
                SpectrogramPreview(
                    width=width,
                    height=height,
                    data=[0] * (width * height),
                    freq_min=freq_min,
                    freq_max=freq_max,
                    duration_sec=total_duration,
                    time_start=time_start,
                    time_end=time_end,
                ),
                1.0,
            )

        start = max(0.0, min(time_start, total_duration))
        end = max(start + 1e-6, min(time_end, total_duration))
        cols_per_sec = (self._matrix_cols - 1) / max(total_duration, 1e-9)
        target_start_col = start * cols_per_sec
        target_end_col = end * cols_per_sec
        start_col = int(np.floor(target_start_col))
        end_col = int(np.ceil(target_end_col)) + 1
        start_col = max(0, min(start_col, self._matrix_cols - 1))
        end_col = max(start_col + 1, min(end_col, self._matrix_cols))

        source = np.asarray(self._matrix[:, start_col:end_col], dtype=np.float32)
        if source.size == 0:
            return (
                SpectrogramPreview(
                    width=width,
                    height=height,
                    data=[0] * (width * height),
                    freq_min=freq_min,
                    freq_max=freq_max,
                    duration_sec=total_duration,
                    time_start=start,
                    time_end=end,
                ),
                1.0,
            )

        effective_min = max(freq_min, settings.freq_min, self._matrix_freq_min)
        effective_max = min(freq_max, settings.preview_freq_max, settings.freq_max, self._matrix_freq_max)
        effective_max = max(effective_max, effective_min * 1.001)
        relative_start_col = target_start_col - float(start_col)
        relative_end_col = target_end_col - float(start_col)
        time_resampled = self._resample_time_matrix(
            source=source,
            width=width,
            relative_start_col=relative_start_col,
            relative_end_col=relative_end_col,
        )
        target_log_freqs = np.geomspace(effective_min, effective_max, height).astype(np.float64)
        composed = self._interpolate_freq_matrix(
            source_freqs=self._matrix_log_freqs,
            source_matrix=time_resampled,
            target_freqs=target_log_freqs,
        )

        computed_amp_reference = self._matrix_max
        reference = computed_amp_reference if amp_reference is None else amp_reference
        normalized = _normalize_magnitude_db(composed, reference_max=reference)
        normalized = np.flipud(normalized)
        normalized = _apply_preview_tone(normalized, settings)
        data = (normalized * 255).astype(np.uint8).flatten().tolist()
        preview = SpectrogramPreview(
            width=width,
            height=height,
            data=data,
            freq_min=effective_min,
            freq_max=effective_max,
            duration_sec=total_duration,
            time_start=start,
            time_end=end,
        )
        return preview, computed_amp_reference

    def _sparse_stft_magnitude(self, audio_segment: np.ndarray, centers: np.ndarray, nperseg: int) -> np.ndarray:
        if centers.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        n = int(min(nperseg, max(256, audio_segment.size)))
        if n & (n - 1) != 0:
            n = 1 << int(np.floor(np.log2(n)))
            n = max(256, n)
        window = np.hanning(n).astype(np.float32)
        half = n // 2
        source = audio_segment.astype(np.float64, copy=False)
        padded = np.pad(source, (half, half), mode="constant")
        starts = centers.astype(np.int64)
        starts = np.clip(starts, 0, max(0, padded.size - n))
        frame_indices = starts[:, None] + np.arange(n, dtype=np.int64)[None, :]
        frames = padded[frame_indices]
        spectra = np.fft.rfft(frames * window[None, :], n=n, axis=1)
        mags = np.abs(spectra).astype(np.float32, copy=False)
        return mags.T

    def _band_weights(self, target_freqs: np.ndarray, band_freq_max: float) -> list[np.ndarray]:
        log2_freqs = np.log2(target_freqs)
        low_center_hz = np.sqrt(20.0 * 200.0)
        mid_center_hz = np.sqrt(200.0 * 2000.0)
        high_center_hz = np.sqrt(2000.0 * max(2000.0, band_freq_max))
        denom_low_mid = max(np.log2(mid_center_hz) - np.log2(low_center_hz), 1e-12)
        denom_mid_high = max(np.log2(high_center_hz) - np.log2(mid_center_hz), 1e-12)
        t1 = np.clip((log2_freqs - np.log2(low_center_hz)) / denom_low_mid, 0.0, 1.0).astype(np.float32)
        t2 = np.clip((log2_freqs - np.log2(mid_center_hz)) / denom_mid_high, 0.0, 1.0).astype(np.float32)
        weight_low = 1.0 - t1
        weight_mid = t1 * (1.0 - t2)
        weight_high = t2
        return [weight_low, weight_mid, weight_high]

    def _resample_time_matrix(
        self,
        source: np.ndarray,
        width: int,
        relative_start_col: float,
        relative_end_col: float,
    ) -> np.ndarray:
        source_width = source.shape[1]
        if source_width == 0:
            return np.zeros((source.shape[0], width), dtype=np.float32)
        if width <= 1:
            sample_x = float(np.clip(relative_start_col, 0.0, max(0.0, source_width - 1)))
            left = int(np.floor(sample_x))
            right = min(source_width - 1, left + 1)
            frac = float(sample_x - left)
            return (source[:, left] * (1.0 - frac) + source[:, right] * frac)[:, None].astype(np.float32)

        relative_start_col = float(np.clip(relative_start_col, 0.0, max(0.0, source_width - 1)))
        relative_end_col = float(np.clip(relative_end_col, relative_start_col, max(0.0, source_width - 1)))
        src_span = max(relative_end_col - relative_start_col, 1e-9)
        source_samples_in_span = src_span + 1.0

        if source_samples_in_span > float(width):
            # ダウンサンプル時はピークを残す。線形補間のみだとピーク欠落が起きやすい。
            boundaries = np.linspace(relative_start_col, relative_end_col, width + 1, dtype=np.float64)
            out = np.empty((source.shape[0], width), dtype=np.float32)
            for i in range(width):
                start = int(np.floor(boundaries[i]))
                end = int(np.ceil(boundaries[i + 1]))
                if end <= start:
                    end = start + 1
                start = max(0, min(start, source_width - 1))
                end = max(start + 1, min(end, source_width))
                out[:, i] = np.max(source[:, start:end], axis=1)
            return out

        src_x = np.arange(source_width, dtype=np.float64)
        dst_x = np.linspace(relative_start_col, relative_end_col, width, dtype=np.float64)
        dst_x = np.clip(dst_x, 0.0, max(0.0, source_width - 1))
        out = np.empty((source.shape[0], width), dtype=np.float32)
        for row in range(source.shape[0]):
            out[row, :] = np.interp(dst_x, src_x, source[row, :]).astype(np.float32)
        return out

    def _interpolate_freq_matrix(
        self,
        source_freqs: np.ndarray,
        source_matrix: np.ndarray,
        target_freqs: np.ndarray,
    ) -> np.ndarray:
        if source_matrix.size == 0:
            cols = source_matrix.shape[1] if source_matrix.ndim == 2 else 0
            return np.zeros((target_freqs.size, cols), dtype=np.float32)
        idx_right = np.searchsorted(source_freqs, target_freqs, side="left")
        idx_right = np.clip(idx_right, 1, source_freqs.size - 1)
        idx_left = idx_right - 1
        left_freqs = source_freqs[idx_left]
        right_freqs = source_freqs[idx_right]
        denom = np.maximum(right_freqs - left_freqs, 1e-12)
        frac = ((target_freqs - left_freqs) / denom).astype(np.float32)
        left_vals = source_matrix[idx_left, :]
        right_vals = source_matrix[idx_right, :]
        out = left_vals * (1.0 - frac[:, None]) + right_vals * frac[:, None]
        return cast(np.ndarray, out)
