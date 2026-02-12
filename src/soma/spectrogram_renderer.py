from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import numpy as np

from soma.analysis import _apply_preview_tone, _normalize_magnitude_db
from soma.models import AnalysisSettings, SpectrogramPreview


class SpectrogramRenderer:
    """Memmap-backed spectrogram renderer."""

    _BASE_HEIGHT = 1024
    _TARGET_COLS_PER_SEC = 84.0
    _MIN_COLS = 2048
    _MAX_COLS = 120_000

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
        preview, ref = self._render_from_global_matrix(
            settings=settings,
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
            width=width,
            height=height,
            amp_reference=stft_amp_reference if stft_amp_reference is not None else cwt_amp_reference,
        )
        return preview, ref, "high"

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

        log2_freqs = np.log2(self._matrix_log_freqs)
        low_center_hz = np.sqrt(20.0 * 200.0)
        mid_center_hz = np.sqrt(200.0 * 2000.0)
        high_center_hz = np.sqrt(2000.0 * max(2000.0, self._matrix_freq_max))
        t1 = np.clip(
            (log2_freqs - np.log2(low_center_hz)) / (np.log2(mid_center_hz) - np.log2(low_center_hz)),
            0.0,
            1.0,
        ).astype(np.float32)
        t2 = np.clip(
            (log2_freqs - np.log2(mid_center_hz)) / (np.log2(high_center_hz) - np.log2(mid_center_hz)),
            0.0,
            1.0,
        ).astype(np.float32)
        weight_low = 1.0 - t1
        weight_mid = t1 * (1.0 - t2)
        weight_high = t2
        weights = [weight_low, weight_mid, weight_high]

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
        start_col = int(np.floor((start / max(total_duration, 1e-9)) * (self._matrix_cols - 1)))
        end_col = int(np.ceil((end / max(total_duration, 1e-9)) * (self._matrix_cols - 1))) + 1
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
        time_resampled = self._resample_time_matrix(source, width)
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
        n = int(min(nperseg, max(256, audio_segment.size)))
        if n & (n - 1) != 0:
            n = 1 << int(np.floor(np.log2(n)))
            n = max(256, n)
        window = np.hanning(n).astype(np.float32)
        half = n // 2
        mags = np.zeros((n // 2 + 1, centers.size), dtype=np.float32)
        source = audio_segment.astype(np.float64, copy=False)
        for col, center in enumerate(centers.tolist()):
            start = int(center) - half
            end = start + n
            if start < 0 or end > audio_segment.size:
                padded = np.zeros(n, dtype=np.float64)
                src_start = max(0, start)
                src_end = min(audio_segment.size, end)
                dst_start = src_start - start
                dst_end = dst_start + (src_end - src_start)
                padded[dst_start:dst_end] = source[src_start:src_end]
                segment = padded
            else:
                segment = source[start:end]
            spectrum = np.fft.rfft(segment * window, n=n)
            mags[:, col] = np.abs(spectrum).astype(np.float32)
        return mags

    def _resample_time_matrix(self, source: np.ndarray, width: int) -> np.ndarray:
        if source.shape[1] == width:
            return source
        if width <= 1:
            return source[:, :1]
        src_x = np.linspace(0.0, 1.0, source.shape[1], dtype=np.float64)
        dst_x = np.linspace(0.0, 1.0, width, dtype=np.float64)
        out = np.empty((source.shape[0], width), dtype=np.float32)
        for row in range(source.shape[0]):
            out[row, :] = np.interp(dst_x, src_x, source[row, :], left=0.0, right=0.0).astype(np.float32)
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
