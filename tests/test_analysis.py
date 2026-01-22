import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("pywt")

from soma.analysis import load_audio, make_spectrogram_preview, snap_trace  # noqa: E402
from soma.models import AnalysisSettings  # noqa: E402


def _write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    wavfile.write(path, sample_rate, data)


def test_load_audio_truncates(tmp_path: Path) -> None:
    path = tmp_path / "test.wav"
    sample_rate = 44100
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    data = (0.5 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
    _write_wav(path, sample_rate, data)

    info, audio = load_audio(path, max_duration_sec=0.5)

    assert info.truncated is True
    assert info.sample_rate == sample_rate
    assert audio.shape[0] <= int(sample_rate * 0.5)


def test_make_spectrogram_preview() -> None:
    sample_rate = 44100
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
    settings = AnalysisSettings()

    preview = make_spectrogram_preview(audio, sample_rate, settings)

    assert preview.width == 768
    assert preview.height == 320
    assert len(preview.data) == preview.width * preview.height
    assert preview.duration_sec > 0.0


def test_snap_trace_returns_points() -> None:
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.8 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
    settings = AnalysisSettings()
    trace = [(0.1, 450.0), (0.2, 445.0), (0.3, 430.0)]

    points = snap_trace(audio, sample_rate, settings, trace)

    assert len(points) == len(trace)
    for point in points:
        assert 400.0 < point.freq < 500.0
        assert 0.0 <= point.amp <= 1.0


def test_snap_trace_decimates_points() -> None:
    sample_rate = 8000
    duration = 0.6
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * math.pi * 330.0 * t)).astype(np.float32)
    settings = AnalysisSettings()
    trace = [(time, 340.0) for time in np.linspace(0.05, 0.55, 200)]

    points = snap_trace(audio, sample_rate, settings, trace, max_points=20)

    assert 1 <= len(points) <= 20

