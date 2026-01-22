from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mido")

from soma.exporter import AudioExportSettings, MpeExportSettings, export_audio, export_mpe  # noqa: E402
from soma.models import Partial, PartialPoint  # noqa: E402


def test_export_mpe(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=0.5),
                PartialPoint(time=0.5, freq=442.0, amp=0.4),
            ],
        )
    ]
    settings = MpeExportSettings(pitch_bend_range=48, amplitude_mapping="velocity")
    path = tmp_path / "project.mid"
    results = export_mpe(partials, path, settings)
    assert len(results) >= 1
    assert results[0].exists()


def test_export_audio(tmp_path: Path) -> None:
    buffer = np.zeros(44100, dtype=np.float32)
    settings = AudioExportSettings(sample_rate=44100, bit_depth=16, output_type="sine")
    path = tmp_path / "audio.wav"
    output = export_audio(path, buffer, settings, 20.0, 20000.0)
    assert output.exists()


def test_export_cv(tmp_path: Path) -> None:
    buffer = np.zeros(100, dtype=np.float32)
    pitch = np.linspace(100.0, 1000.0, 100, dtype=np.float32)
    amp = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    settings = AudioExportSettings(sample_rate=100, bit_depth=16, output_type="cv")
    path = tmp_path / "cv.wav"
    output = export_audio(path, buffer, settings, 20.0, 20000.0, pitch_buffer=pitch, amp_buffer=amp)
    assert output.exists()
