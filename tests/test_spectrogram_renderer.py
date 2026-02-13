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
        assert quality in {"high", "local"}
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


def test_adjacent_tiles_are_continuous_for_steady_tone() -> None:
    sample_rate = 44100
    duration_sec = 3.0
    t = np.arange(int(sample_rate * duration_sec), dtype=np.float64) / float(sample_rate)
    audio = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    settings = AnalysisSettings()

    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        left, _, _ = renderer.render_tile(
            settings=settings,
            time_start=1.0,
            time_end=1.5,
            freq_min=20.0,
            freq_max=12000.0,
            width=1024,
            height=256,
        )
        right, _, _ = renderer.render_tile(
            settings=settings,
            time_start=1.5,
            time_end=2.0,
            freq_min=20.0,
            freq_max=12000.0,
            width=1024,
            height=256,
        )
        left_data = np.asarray(left.data, dtype=np.float32).reshape((left.height, left.width))
        right_data = np.asarray(right.data, dtype=np.float32).reshape((right.height, right.width))
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
            settings=settings,
            time_start=start,
            time_end=end,
            freq_min=20.0,
            freq_max=12000.0,
            width=800,
            height=192,
        )
        left, _, _ = renderer.render_tile(
            settings=settings,
            time_start=start,
            time_end=split,
            freq_min=20.0,
            freq_max=12000.0,
            width=400,
            height=192,
        )
        right, _, _ = renderer.render_tile(
            settings=settings,
            time_start=split,
            time_end=end,
            freq_min=20.0,
            freq_max=12000.0,
            width=400,
            height=192,
        )
        full_data = np.asarray(full.data, dtype=np.float32).reshape((full.height, full.width))
        left_data = np.asarray(left.data, dtype=np.float32).reshape((left.height, left.width))
        right_data = np.asarray(right.data, dtype=np.float32).reshape((right.height, right.width))
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
            settings=settings,
            time_start=0.2,
            time_end=0.7,
            freq_min=60.0,
            freq_max=6000.0,
            width=256,
            height=128,
        )
        assert quality1 == "local"

        _preview2, _ref2, quality2 = renderer.render_tile(
            settings=settings,
            time_start=0.0,
            time_end=1.0,
            freq_min=60.0,
            freq_max=6000.0,
            width=80,
            height=128,
        )
        assert quality2 == "local"

        _preview3, _ref3, quality3 = renderer.render_tile(
            settings=settings,
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
