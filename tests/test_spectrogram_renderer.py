from __future__ import annotations

import numpy as np

from soma.models import AnalysisSettings
from soma.spectrogram_renderer import SpectrogramRenderer


def test_render_tile_returns_preview_with_expected_shape() -> None:
    sample_rate = 8000
    duration_sec = 1.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    settings = AnalysisSettings()

    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        preview, amp_ref, quality = renderer.render_tile(
            settings=settings,
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
        assert quality == "high"
    finally:
        renderer.close()


def test_render_overview_with_empty_audio_returns_black_frame() -> None:
    settings = AnalysisSettings()
    renderer = SpectrogramRenderer(audio=np.zeros(0, dtype=np.float32), sample_rate=44100)
    try:
        preview, amp_ref = renderer.render_overview(settings=settings, width=64, height=32)
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
