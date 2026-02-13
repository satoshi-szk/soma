from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from soma.models import AnalysisSettings, SpectrogramPreview, SpectrogramSettings
from soma.spectrogram_renderer import SpectrogramRenderer


def _preview_matrix(preview: SpectrogramPreview) -> np.ndarray:
    data = preview.data
    width = preview.width
    height = preview.height
    if isinstance(data, bytes):
        return np.frombuffer(data, dtype=np.uint8).astype(np.float32).reshape((height, width))
    return np.asarray(data, dtype=np.float32).reshape((height, width))


def test_render_tile_returns_preview_with_expected_shape() -> None:
    sample_rate = 8000
    duration_sec = 1.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    settings = AnalysisSettings()

    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        preview, amp_ref, quality = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=0.0,
            time_end=0.5,
            freq_min=80.0,
            freq_max=2000.0,
            width=256,
            height=128,
        )
        assert preview.width == 256
        assert preview.height == 128
        assert len(preview.data) == 256 * 128
        assert amp_ref > 0.0
        assert quality in {"high", "local"}
    finally:
        renderer.close()


def test_reassigned_render_uses_same_method_for_overview_and_tile() -> None:
    sample_rate = 16000
    duration_sec = 1.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    settings = AnalysisSettings(spectrogram=SpectrogramSettings(method="reassigned_stft", reassigned_ref_power=1e-6))

    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        overview, overview_ref = renderer.render_overview(settings=settings.spectrogram, width=128, height=64)
        tile, tile_ref, quality = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=0.0,
            time_end=0.5,
            freq_min=40.0,
            freq_max=4000.0,
            width=128,
            height=64,
        )
        assert overview.width == 128
        assert overview.height == 64
        assert len(overview.data) == 128 * 64
        assert overview_ref > 0.0
        assert tile.width == 128
        assert tile.height == 64
        assert len(tile.data) == 128 * 64
        assert tile_ref > 0.0
        assert quality in {"high", "local"}
    finally:
        renderer.close()


def test_render_overview_with_empty_audio_returns_black_frame() -> None:
    settings = AnalysisSettings()
    renderer = SpectrogramRenderer(audio=np.zeros(0, dtype=np.float32), sample_rate=44100)
    try:
        preview, amp_ref = renderer.render_overview(settings=settings.spectrogram, width=64, height=32)
        assert preview.width == 64
        assert preview.height == 32
        assert set(preview.data) == {0}
        assert amp_ref == 1.0
    finally:
        renderer.close()


def test_close_is_idempotent() -> None:
    renderer = SpectrogramRenderer(audio=np.zeros(128, dtype=np.float32), sample_rate=44100)
    renderer.close()
    renderer.close()


def test_adjacent_tiles_are_continuous_for_steady_tone() -> None:
    sample_rate = 44100
    duration_sec = 3.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    settings = AnalysisSettings()

    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        left, _, _ = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=1.0,
            time_end=1.5,
            freq_min=20.0,
            freq_max=12000.0,
            width=1024,
            height=256,
        )
        right, _, _ = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=1.5,
            time_end=2.0,
            freq_min=20.0,
            freq_max=12000.0,
            width=1024,
            height=256,
        )
        left_data = _preview_matrix(left)
        right_data = _preview_matrix(right)
        seam_delta = np.abs(left_data[:, -1] - right_data[:, 0])
        assert float(np.mean(seam_delta)) < 6.0
    finally:
        renderer.close()


def test_split_tiles_match_single_render_for_fractional_boundaries() -> None:
    sample_rate = 44100
    duration_sec = 3.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.2 * np.sin(2.0 * np.pi * (180.0 + 120.0 * t) * t)).astype(np.float32)
    settings = AnalysisSettings()
    start = 0.73
    split = 1.19
    end = 1.61

    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        full, _, _ = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=start,
            time_end=end,
            freq_min=20.0,
            freq_max=12000.0,
            width=800,
            height=192,
        )
        left, _, _ = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=start,
            time_end=split,
            freq_min=20.0,
            freq_max=12000.0,
            width=400,
            height=192,
        )
        right, _, _ = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=split,
            time_end=end,
            freq_min=20.0,
            freq_max=12000.0,
            width=400,
            height=192,
        )
        full_data = _preview_matrix(full)
        left_data = _preview_matrix(left)
        right_data = _preview_matrix(right)
        combined = np.concatenate([left_data, right_data], axis=1)
        seam_delta = np.abs(left_data[:, -1] - right_data[:, 0])
        assert float(np.mean(seam_delta)) < 4.0
        assert float(np.mean(np.abs(combined - full_data))) < 5.0
    finally:
        renderer.close()


def test_render_tile_switches_to_local_mode_with_hysteresis() -> None:
    sample_rate = 16000
    duration_sec = 2.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.3 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    settings = AnalysisSettings()

    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        _preview1, _ref1, quality1 = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=0.2,
            time_end=0.7,
            freq_min=60.0,
            freq_max=6000.0,
            width=256,
            height=128,
        )
        assert quality1 == "local"

        _preview2, _ref2, quality2 = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=0.0,
            time_end=1.0,
            freq_min=60.0,
            freq_max=6000.0,
            width=80,
            height=128,
        )
        assert quality2 == "local"

        _preview3, _ref3, quality3 = renderer.render_tile(
            settings=settings.spectrogram,
            time_start=0.0,
            time_end=1.0,
            freq_min=60.0,
            freq_max=6000.0,
            width=60,
            height=128,
        )
        assert quality3 == "high"
    finally:
        renderer.close()


def test_sparse_stft_supports_zero_padding_nfft() -> None:
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float64) / float(sample_rate)
    audio = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        centers = np.array([100, 300, 500], dtype=np.int64)
        base = renderer._sparse_stft_magnitude(audio, centers=centers, nperseg=256)
        padded = renderer._sparse_stft_magnitude(audio, centers=centers, nperseg=256, nfft=4096)
        assert base.shape[0] == 129
        assert padded.shape[0] == 2049
        assert padded.shape[1] == base.shape[1]
    finally:
        renderer.close()


def test_compute_local_band_cols_uses_rx_hop_and_pixel_hop() -> None:
    renderer = SpectrogramRenderer(audio=np.zeros(4096, dtype=np.float32), sample_rate=16000)
    try:
        # pixel_hop の方が小さいので、band ごとの差は吸収されて width 近傍になる。
        cols_low = renderer._compute_local_band_cols(view_samples=8192.0, width=1024, nperseg=4096)
        cols_high = renderer._compute_local_band_cols(view_samples=8192.0, width=1024, nperseg=256)
        assert cols_low == 1025
        assert cols_high == 1025

        # RX 基準 hop が選ばれるケースでは列数が増えるが、上限でクランプされる。
        cols_capped = renderer._compute_local_band_cols(view_samples=1_048_576.0, width=1024, nperseg=256)
        assert cols_capped == renderer._LOCAL_MAX_INTERNAL_COLS
    finally:
        renderer.close()


def test_render_tile_is_safe_under_concurrent_requests() -> None:
    sample_rate = 44100
    duration_sec = 2.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.25 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    settings = AnalysisSettings()
    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        def render_once(index: int) -> tuple[int, int, str]:
            start = 0.1 + (index % 6) * 0.1
            end = min(start + 0.35, duration_sec)
            preview, _amp_ref, quality = renderer.render_tile(
                settings=settings.spectrogram,
                time_start=start,
                time_end=end,
                freq_min=40.0,
                freq_max=9000.0,
                width=320,
                height=128,
            )
            return preview.width, preview.height, quality

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(render_once, range(40)))

        assert all(width == 320 and height == 128 for width, height, _quality in results)
        assert all(quality in {"high", "local"} for _width, _height, quality in results)
    finally:
        renderer.close()
