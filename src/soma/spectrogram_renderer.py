from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from soma.analysis import _apply_preview_tone, _normalize_magnitude_db
from soma.models import AnalysisSettings, SpectrogramPreview


class SpectrogramRenderer:
    """Memmap-backed spectrogram renderer."""

    def __init__(self, audio: np.ndarray, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._length = int(audio.shape[0])
        self._temp_dir = tempfile.TemporaryDirectory(prefix="soma-spec-")
        self._audio_path = Path(self._temp_dir.name) / "audio.dat"
        self._closed = False
        try:
            write_map = np.memmap(self._audio_path, dtype=np.float32, mode="w+", shape=(self._length,))
            write_map[:] = np.asarray(audio, dtype=np.float32)
            write_map.flush()
            del write_map
            self._audio = np.memmap(self._audio_path, dtype=np.float32, mode="r", shape=(self._length,))
        except Exception:
            self._temp_dir.cleanup()
            raise

    def close(self) -> None:
        if self._closed:
            return
        if hasattr(self, "_audio"):
            del self._audio
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
        return self._render_multires_preview(
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
        preview, ref = self._render_multires_preview(
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

    def _render_multires_preview(
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
        total_duration = self._length / float(self._sample_rate) if self._sample_rate > 0 else 0.0
        if self._length == 0 or time_end <= time_start:
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
        start_idx = int(start * self._sample_rate)
        end_idx = int(end * self._sample_rate)
        segment = np.asarray(self._audio[start_idx:end_idx], dtype=np.float32)
        if segment.size < 32:
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

        effective_min = max(freq_min, settings.freq_min, 20.0)
        effective_max = min(freq_max, settings.preview_freq_max, settings.freq_max, self._sample_rate * 0.5)
        effective_max = max(effective_max, effective_min * 1.001)
        log_freqs = np.geomspace(effective_min, effective_max, height).astype(np.float64)

        bands = [
            (20.0, 200.0, 4096),
            (200.0, 2000.0, 1024),
            (2000.0, effective_max, 256),
        ]
        composed = np.zeros((height, width), dtype=np.float32)
        for low_hz, high_hz, nperseg in bands:
            band_lo = max(effective_min, low_hz)
            band_hi = min(effective_max, high_hz)
            if band_hi <= band_lo:
                continue
            band_mag = self._sparse_stft_magnitude(segment, width=width, nperseg=nperseg)
            if band_mag.size == 0:
                continue
            fft_freqs = np.fft.rfftfreq(band_mag.shape[0] * 2 - 2, d=1.0 / float(self._sample_rate)).astype(np.float64)
            resized = np.empty((height, width), dtype=np.float32)
            for col in range(width):
                resized[:, col] = np.interp(log_freqs, fft_freqs, band_mag[:, col], left=0.0, right=0.0).astype(
                    np.float32
                )
            mask = (log_freqs >= band_lo) & (log_freqs <= band_hi)
            composed[mask, :] = resized[mask, :]

        computed_amp_reference = float(np.max(composed)) if composed.size else 1.0
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

    def _sparse_stft_magnitude(self, audio_segment: np.ndarray, width: int, nperseg: int) -> np.ndarray:
        n = int(min(nperseg, max(256, audio_segment.size)))
        if n & (n - 1) != 0:
            n = 1 << int(np.floor(np.log2(n)))
            n = max(256, n)
        window = np.hanning(n).astype(np.float32)
        frame_centers = np.linspace(0, audio_segment.size - 1, width, dtype=np.int64)
        half = n // 2
        mags = np.zeros((n // 2 + 1, width), dtype=np.float32)
        source = audio_segment.astype(np.float64, copy=False)
        for col, center in enumerate(frame_centers.tolist()):
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
