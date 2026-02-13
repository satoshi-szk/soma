from __future__ import annotations

import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import cast

import librosa
import numpy as np

from soma.analysis import _apply_preview_tone, _normalize_magnitude_db
from soma.models import SpectrogramPreview, SpectrogramSettings


class SpectrogramRenderer:
    """Memmap-backed spectrogram renderer."""

    _BASE_HEIGHT = 1024
    _TARGET_COLS_PER_SEC = 84.0
    _LOCAL_ENTER_MULTIPLIER = 1.10
    _LOCAL_EXIT_MULTIPLIER = 0.90
    _LOCAL_CACHE_MAX_ITEMS = 24
    _LOCAL_MAX_INTERNAL_COLS = 4096
    _LOCAL_TIME_QUANTUM_SEC = 0.001
    _LOCAL_FREQ_QUANTUM_HZ = 0.1
    _MIN_COLS = 2048
    _MAX_COLS = 320_000
    _MULTIRES_BANDS_BASE: tuple[tuple[float, float, int, int], ...] = (
        (20.0, 200.0, 8192, 512),
        (200.0, 2000.0, 2048, 128),
        (2000.0, 0.0, 512, 32),  # high band の上限は実行時に Nyquist で置換
    )
    _WINDOW_SCALE_OPTIONS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        global_blend_octaves: float = 1.0,
        global_window_size_scale: float = 1.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._length = int(audio.shape[0])
        blend = float(global_blend_octaves)
        if not np.isfinite(blend) or blend <= 0.0:
            blend = 1.0
        self._global_blend_octaves = max(0.01, min(6.0, blend))
        self._global_window_size_scale = self._resolve_window_scale(global_window_size_scale)
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
            self._mode_lock = threading.Lock()
            self._local_cache_lock = threading.Lock()
            self._reassigned_reference_lock = threading.Lock()
            self._prefer_local_mode = False
            self._local_cache: OrderedDict[
                tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int],
                tuple[SpectrogramPreview, float],
            ] = OrderedDict()
            self._reassigned_reference_cache: dict[tuple[int, int, int, int, int], float] = {}
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
        settings: SpectrogramSettings,
        width: int,
        height: int,
        stft_amp_reference: float | None = None,
    ) -> tuple[SpectrogramPreview, float]:
        duration = self._length / float(self._sample_rate) if self._sample_rate > 0 else 0.0
        if settings.method == "reassigned_stft":
            return self._render_reassigned_tile(
                settings=settings,
                time_start=0.0,
                time_end=duration,
                freq_min=settings.freq_min,
                freq_max=min(settings.preview_freq_max, settings.freq_max),
                width=width,
                height=height,
                amp_reference=stft_amp_reference,
            )
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
        settings: SpectrogramSettings,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
        stft_amp_reference: float | None = None,
    ) -> tuple[SpectrogramPreview, float, str]:
        time_span = max(time_end - time_start, 1e-9)
        required_cols_per_sec = width / time_span
        use_local = self._should_use_local_mode(required_cols_per_sec)
        amp_reference = stft_amp_reference

        if settings.method == "reassigned_stft":
            preview, ref = self._render_reassigned_tile(
                settings=settings,
                time_start=time_start,
                time_end=time_end,
                freq_min=freq_min,
                freq_max=freq_max,
                width=width,
                height=height,
                amp_reference=amp_reference,
                use_cache=use_local,
            )
            return preview, ref, "local" if use_local else "high"

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
        with self._mode_lock:
            if self._prefer_local_mode:
                if required_cols_per_sec < exit_threshold:
                    self._prefer_local_mode = False
            elif required_cols_per_sec > enter_threshold:
                self._prefer_local_mode = True
            return self._prefer_local_mode

    def _render_local_tile(
        self,
        settings: SpectrogramSettings,
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
        with self._local_cache_lock:
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

        band_freq_max = max(effective_max, nyquist)
        bands = self._multires_band_specs(band_freq_max)
        max_nperseg = max(band[2] for band in bands)
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

        target_log_freqs = np.geomspace(effective_min, effective_max, height).astype(np.float64)
        weights = self._band_weights(target_log_freqs, band_freq_max, settings.multires_blend_octaves)
        composed = np.zeros((height, width), dtype=np.float32)
        view_samples = max((end - start) * float(self._sample_rate), 1.0)

        for band_index, (_band_lo, _band_hi, nperseg, nfft, hop_samples) in enumerate(bands):
            band_cols = self._compute_local_band_cols(
                view_samples=view_samples,
                width=width,
                nperseg=nperseg,
                hop_samples=hop_samples,
            )
            if band_cols <= 1:
                center_times = np.array([0.5 * (start + end)], dtype=np.float64)
            else:
                center_times = np.linspace(start, end, band_cols, dtype=np.float64)
            centers = np.rint(center_times * float(self._sample_rate)).astype(np.int64) - np.int64(ext_start_idx)
            band_mag = self._sparse_stft_magnitude(local_audio, centers=centers, nperseg=nperseg, nfft=nfft)
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
            if band_cols != width:
                band_image = self._resample_time_matrix(
                    source=band_image,
                    width=width,
                    relative_start_col=0.0,
                    relative_end_col=float(max(0, band_cols - 1)),
                )
            composed += band_image * weights[band_index][:, None]

        computed_amp_reference = self._matrix_max
        reference = computed_amp_reference if amp_reference is None else amp_reference
        normalized = _normalize_magnitude_db(composed, reference_max=reference)
        normalized = np.flipud(normalized)
        normalized = _apply_preview_tone(normalized, settings)
        data = (normalized * 255).astype(np.uint8, copy=False).tobytes()
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

    def _render_reassigned_tile(
        self,
        settings: SpectrogramSettings,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
        amp_reference: float | None,
        use_cache: bool = False,
    ) -> tuple[SpectrogramPreview, float]:
        cache_key: tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int] | None = None
        if use_cache:
            cache_key = self._build_local_cache_key(
                settings=settings,
                time_start=time_start,
                time_end=time_end,
                freq_min=freq_min,
                freq_max=freq_max,
                width=width,
                height=height,
            )
            with self._local_cache_lock:
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
            if use_cache and cache_key is not None:
                self._store_local_cache(cache_key, result)
            return result

        start = max(0.0, min(time_start, total_duration))
        end = max(start + 1e-6, min(time_end, total_duration))
        nyquist = float(self._sample_rate) * 0.5
        effective_min = max(freq_min, settings.freq_min, 20.0)
        effective_max = min(freq_max, settings.preview_freq_max, settings.freq_max, nyquist)
        effective_max = max(effective_max, effective_min * 1.001)

        max_nperseg = max(band[2] for band in self._multires_band_specs(float(effective_max)))
        margin_sec = max((max_nperseg / float(self._sample_rate)) * 0.5, (end - start) * 0.05, 0.01)
        ext_start = max(0.0, start - margin_sec)
        ext_end = min(total_duration, end + margin_sec)
        ext_start_sample = int(np.floor(ext_start * self._sample_rate))
        ext_end_sample = int(np.ceil(ext_end * self._sample_rate))
        ext_end_sample = max(ext_end_sample, ext_start_sample + 1)
        segment = np.asarray(self._audio[ext_start_sample:ext_end_sample], dtype=np.float32)
        if segment.size < 32:
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
            if use_cache and cache_key is not None:
                self._store_local_cache(cache_key, result)
            return result

        freq_edges = np.geomspace(effective_min, effective_max, height + 1).astype(np.float64)
        freq_centers = np.sqrt(freq_edges[:-1] * freq_edges[1:])
        time_edges = np.linspace(start, end, width + 1, dtype=np.float64)
        composed_map = self._compute_reassigned_composed_map(
            settings=settings,
            segment=segment,
            time_start=ext_start,
            time_end=ext_end,
            view_time_start=start,
            view_time_end=end,
            freq_edges=freq_edges,
            freq_centers=freq_centers,
            time_edges=time_edges,
            height=height,
            width=width,
            effective_min=effective_min,
            effective_max=effective_max,
        )

        global_reference = self._get_reassigned_global_reference(settings)
        reference = global_reference if amp_reference is None else amp_reference
        normalized = _normalize_magnitude_db(composed_map, reference_max=reference)
        normalized = np.flipud(normalized)
        normalized = _apply_preview_tone(normalized, settings)
        data = (normalized * 255).astype(np.uint8, copy=False).tobytes()
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
        result = (preview, global_reference)
        if use_cache and cache_key is not None:
            self._store_local_cache(cache_key, result)
        return result

    def _get_reassigned_global_reference(self, settings: SpectrogramSettings) -> float:
        key = (
            int(round(settings.freq_min * 10.0)),
            int(round(settings.freq_max * 10.0)),
            int(round(settings.preview_freq_max * 10.0)),
            int(round(settings.multires_blend_octaves * 1000.0)),
            int(round(settings.reassigned_ref_power * 1_000_000_000.0)),
        )
        with self._reassigned_reference_lock:
            cached = self._reassigned_reference_cache.get(key)
            if cached is not None:
                return cached

        total_duration = self._length / float(self._sample_rate) if self._sample_rate > 0 else 0.0
        if total_duration <= 0.0 or self._length == 0 or self._sample_rate <= 0:
            return 1.0
        nyquist = float(self._sample_rate) * 0.5
        effective_min = max(settings.freq_min, 20.0)
        effective_max = min(settings.preview_freq_max, settings.freq_max, nyquist)
        effective_max = max(effective_max, effective_min * 1.001)

        global_width = min(2048, max(512, int(np.ceil(total_duration * self._TARGET_COLS_PER_SEC))))
        global_height = min(512, self._BASE_HEIGHT)
        segment = np.asarray(self._audio, dtype=np.float32)
        freq_edges = np.geomspace(effective_min, effective_max, global_height + 1).astype(np.float64)
        freq_centers = np.sqrt(freq_edges[:-1] * freq_edges[1:])
        time_edges = np.linspace(0.0, total_duration, global_width + 1, dtype=np.float64)
        composed_map = self._compute_reassigned_composed_map(
            settings=settings,
            segment=segment,
            time_start=0.0,
            time_end=total_duration,
            view_time_start=0.0,
            view_time_end=total_duration,
            freq_edges=freq_edges,
            freq_centers=freq_centers,
            time_edges=time_edges,
            height=global_height,
            width=global_width,
            effective_min=effective_min,
            effective_max=effective_max,
        )
        reference = float(np.max(composed_map)) if composed_map.size else 1.0
        if reference <= 0.0:
            reference = 1.0
        with self._reassigned_reference_lock:
            self._reassigned_reference_cache[key] = reference
        return reference

    def _compute_reassigned_composed_map(
        self,
        settings: SpectrogramSettings,
        segment: np.ndarray,
        time_start: float,
        time_end: float,
        view_time_start: float,
        view_time_end: float,
        freq_edges: np.ndarray,
        freq_centers: np.ndarray,
        time_edges: np.ndarray,
        height: int,
        width: int,
        effective_min: float,
        effective_max: float,
    ) -> np.ndarray:
        composed_map = np.zeros((height, width), dtype=np.float32)
        bands = self._multires_band_specs(float(effective_max))
        weights = self._band_weights(freq_centers, float(effective_max), settings.multires_blend_octaves)
        ref_power = max(0.0, float(settings.reassigned_ref_power))
        log_min = float(np.log(effective_min))
        log_span = max(float(np.log(effective_max)) - log_min, 1e-9)
        view_time_span = max(view_time_end - view_time_start, 1e-9)

        for band_index, (_band_lo, _band_hi, nperseg, _nfft, hop_samples) in enumerate(bands):
            n_fft = max(256, int(nperseg))
            if n_fft & (n_fft - 1) != 0:
                n_fft = 1 << int(np.floor(np.log2(n_fft)))
                n_fft = max(256, n_fft)
            hop_length = min(max(1, int(hop_samples)), n_fft)
            band_segment = segment
            if band_segment.size < n_fft:
                band_segment = np.pad(band_segment, (0, n_fft - band_segment.size), mode="constant")

            freqs, times, d_reassigned = librosa.reassigned_spectrogram(
                y=band_segment,
                sr=self._sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window="hann",
                center=False,
                reassign_frequencies=True,
                reassign_times=True,
                clip=True,
                ref_power=ref_power,
                fill_nan=True,
            )
            mags = np.abs(d_reassigned).astype(np.float32)
            if mags.ndim != 2 or mags.shape[1] == 0:
                continue
            frame_count = mags.shape[1]

            band_map = np.zeros((height, width), dtype=np.float32)
            for col in range(mags.shape[1]):
                col_freqs = freqs[:, col]
                col_times = times[:, col] + time_start
                col_mags = mags[:, col]
                valid = (
                    np.isfinite(col_freqs)
                    & np.isfinite(col_times)
                    & np.isfinite(col_mags)
                    & (col_freqs >= effective_min)
                    & (col_freqs <= effective_max)
                    & (col_times >= view_time_start)
                    & (col_times <= view_time_end)
                )
                if not np.any(valid):
                    continue
                valid_freqs = col_freqs[valid].astype(np.float64)
                valid_times = col_times[valid].astype(np.float64)
                valid_mags = col_mags[valid].astype(np.float32)

                row_pos = ((np.log(valid_freqs) - log_min) / log_span) * float(height - 1)
                col_pos = ((valid_times - view_time_start) / view_time_span) * float(width - 1)
                row_pos = np.clip(row_pos, 0.0, float(height - 1))
                col_pos = np.clip(col_pos, 0.0, float(width - 1))

                row0 = np.floor(row_pos).astype(np.int32)
                col0 = np.floor(col_pos).astype(np.int32)
                row1 = np.minimum(row0 + 1, height - 1)
                col1 = np.minimum(col0 + 1, width - 1)
                row_alpha = (row_pos - row0).astype(np.float32)
                col_alpha = (col_pos - col0).astype(np.float32)
                w00 = (1.0 - row_alpha) * (1.0 - col_alpha)
                w01 = (1.0 - row_alpha) * col_alpha
                w10 = row_alpha * (1.0 - col_alpha)
                w11 = row_alpha * col_alpha
                np.add.at(band_map, (row0, col0), valid_mags * w00)
                np.add.at(band_map, (row0, col1), valid_mags * w01)
                np.add.at(band_map, (row1, col0), valid_mags * w10)
                np.add.at(band_map, (row1, col1), valid_mags * w11)

            # フレーム密度で正規化して、ズーム（時間幅）で明るさが変わるのを抑える。
            density = float(width) / float(max(1, frame_count))
            band_map *= density
            composed_map += band_map * weights[band_index][:, None]
        return composed_map

    def _build_local_cache_key(
        self,
        settings: SpectrogramSettings,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int]:
        method = 1 if settings.method == "reassigned_stft" else 0
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
            int(round(settings.multires_blend_octaves * 1000.0)),
            int(round(settings.multires_window_size_scale * 1000.0)),
            method,
            int(round(settings.reassigned_ref_power * 1_000_000_000.0)),
        )

    def _store_local_cache(
        self,
        cache_key: tuple[int, int, int, int, int, int, int, int, int, int, int, int, int, int],
        value: tuple[SpectrogramPreview, float],
    ) -> None:
        with self._local_cache_lock:
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

        bands = self._multires_band_specs(self._matrix_freq_max)

        weights = self._band_weights(self._matrix_log_freqs, self._matrix_freq_max, self._global_blend_octaves)

        chunk_cols = 256
        for chunk_start in range(0, self._matrix_cols, chunk_cols):
            chunk_end = min(self._matrix_cols, chunk_start + chunk_cols)
            chunk_centers = centers[chunk_start:chunk_end]
            composed_chunk = np.zeros((self._BASE_HEIGHT, chunk_end - chunk_start), dtype=np.float32)
            for band_index, (_band_lo, _band_hi, nperseg, nfft, _hop_samples) in enumerate(bands):
                band_mag = self._sparse_stft_magnitude(self._audio, centers=chunk_centers, nperseg=nperseg, nfft=nfft)
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
        settings: SpectrogramSettings,
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
        data = (normalized * 255).astype(np.uint8, copy=False).tobytes()
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

    def _sparse_stft_magnitude(
        self,
        audio_segment: np.ndarray,
        centers: np.ndarray,
        nperseg: int,
        nfft: int | None = None,
    ) -> np.ndarray:
        if centers.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        n = int(min(nperseg, max(256, audio_segment.size)))
        if n & (n - 1) != 0:
            n = 1 << int(np.floor(np.log2(n)))
            n = max(256, n)
        fft_size = n
        if nfft is not None:
            fft_size = max(int(nfft), n)
        window = np.hanning(n).astype(np.float32)
        half = n // 2
        source = audio_segment.astype(np.float64, copy=False)
        padded = np.pad(source, (half, half), mode="constant")
        starts = centers.astype(np.int64)
        starts = np.clip(starts, 0, max(0, padded.size - n))
        frame_indices = starts[:, None] + np.arange(n, dtype=np.int64)[None, :]
        frames = padded[frame_indices]
        spectra = np.fft.rfft(frames * window[None, :], n=fft_size, axis=1)
        mags = np.abs(spectra).astype(np.float32, copy=False)
        return mags.T

    def _compute_local_band_cols(
        self,
        view_samples: float,
        width: int,
        nperseg: int,
        hop_samples: float | None = None,
    ) -> int:
        if width <= 1:
            return 1
        pixel_hop = max(view_samples / float(width), 1.0)
        rx_hop = max(float(nperseg) / 8.0, 1.0) if hop_samples is None else max(float(hop_samples), 1.0)
        hop = min(rx_hop, pixel_hop)
        cols = int(np.floor(view_samples / hop)) + 1
        cols = max(width, cols)
        return min(cols, max(width, self._LOCAL_MAX_INTERNAL_COLS))

    def _multires_band_specs(self, high_band_max: float) -> list[tuple[float, float, int, int, int]]:
        bands: list[tuple[float, float, int, int, int]] = []
        for band_lo, band_hi, nperseg, hop_samples in self._MULTIRES_BANDS_BASE:
            resolved_hi = high_band_max if band_hi <= 0.0 else band_hi
            scaled_nperseg = int(round(float(nperseg) * self._global_window_size_scale))
            scaled_nperseg = max(256, scaled_nperseg)
            bands.append((band_lo, resolved_hi, scaled_nperseg, 4096, hop_samples))
        return bands

    def _resolve_window_scale(self, value: float) -> float:
        if not np.isfinite(value):
            return 1.0
        return min(self._WINDOW_SCALE_OPTIONS, key=lambda candidate: abs(candidate - float(value)))

    def _band_weights(
        self,
        target_freqs: np.ndarray,
        band_freq_max: float,
        blend_octaves: float,
    ) -> list[np.ndarray]:
        log2_freqs = np.log2(target_freqs)
        blend = float(blend_octaves)
        if not np.isfinite(blend) or blend <= 0.0:
            blend = 1.0
        blend = max(0.01, min(6.0, blend))
        half = blend * 0.5
        edge_low_mid = np.log2(np.sqrt(200.0 * 200.0))
        edge_mid_high = np.log2(np.sqrt(2000.0 * 2000.0))

        def _smoothstep(x: np.ndarray) -> np.ndarray:
            clipped = np.clip(x, 0.0, 1.0).astype(np.float32)
            return np.asarray(clipped * clipped * (3.0 - 2.0 * clipped), dtype=np.float32)

        t1 = _smoothstep((log2_freqs - (edge_low_mid - half)) / blend)
        t2 = _smoothstep((log2_freqs - (edge_mid_high - half)) / blend)
        weight_low = 1.0 - t1
        weight_mid = t1 * (1.0 - t2)
        weight_high = t2
        if band_freq_max < 2000.0:
            weight_high = np.zeros_like(weight_high)
            total = np.maximum(weight_low + weight_mid, 1e-12)
            weight_low = weight_low / total
            weight_mid = weight_mid / total
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
