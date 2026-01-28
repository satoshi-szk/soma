from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("mido")

from soma.exporter import (  # noqa: E402
    AudioExportSettings,
    MonophonicExportSettings,
    MpeExportSettings,
    MultiTrackExportSettings,
    export_audio,
    export_monophonic_midi,
    export_mpe,
    export_multitrack_midi,
)
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


def test_export_mpe_retrigger(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=0.8),
                PartialPoint(time=0.5, freq=880.0, amp=0.6),
            ],
        )
    ]
    settings = MpeExportSettings(pitch_bend_range=1, amplitude_mapping="velocity")
    path = tmp_path / "retrigger.mid"
    results = export_mpe(partials, path, settings)
    midi = pytest.importorskip("mido").MidiFile(results[0])
    note_ons = [msg for msg in midi.tracks[1] if msg.type == "note_on" and msg.velocity > 0]
    assert len(note_ons) == 2


def test_export_mpe_amp_db_mapping(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=0.2),
                PartialPoint(time=0.5, freq=440.0, amp=0.1),
            ],
        )
    ]
    settings = MpeExportSettings(pitch_bend_range=48, amplitude_mapping="pressure")
    path = tmp_path / "amp.mid"
    results = export_mpe(partials, path, settings)
    midi = pytest.importorskip("mido").MidiFile(results[0])
    aftertouch = [msg.value for msg in midi.tracks[1] if msg.type == "aftertouch"]
    assert len(aftertouch) >= 2
    assert aftertouch[0] == 127
    assert aftertouch[1] == 1


def test_export_audio(tmp_path: Path) -> None:
    buffer = np.zeros(44100, dtype=np.float32)
    settings = AudioExportSettings(sample_rate=44100, bit_depth=16, output_type="sine")
    path = tmp_path / "audio.wav"
    output = export_audio(path, buffer, settings, 20.0, 20000.0)
    assert output.exists()


def test_export_multitrack_midi(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=0.5),
                PartialPoint(time=1.0, freq=442.0, amp=0.4),
            ],
        ),
        Partial(
            id="p2",
            points=[
                PartialPoint(time=0.5, freq=660.0, amp=0.5),
                PartialPoint(time=1.5, freq=661.0, amp=0.4),
            ],
        ),
    ]
    settings = MultiTrackExportSettings(pitch_bend_range=48, amplitude_mapping="velocity")
    path = tmp_path / "multitrack.mid"
    results = export_multitrack_midi(partials, path, settings)
    midi = pytest.importorskip("mido").MidiFile(results[0])
    assert len(midi.tracks) == 2


def test_export_monophonic_midi(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=0.5),
                PartialPoint(time=1.0, freq=441.0, amp=0.4),
            ],
        ),
        Partial(
            id="p2",
            points=[
                PartialPoint(time=0.5, freq=660.0, amp=0.5),
                PartialPoint(time=1.5, freq=659.0, amp=0.4),
            ],
        ),
    ]
    settings = MonophonicExportSettings(pitch_bend_range=48, amplitude_mapping="velocity")
    path = tmp_path / "mono.mid"
    results = export_monophonic_midi(partials, path, settings)
    midi = pytest.importorskip("mido").MidiFile(results[0])
    assert len(midi.tracks) == 1


def test_export_cv(tmp_path: Path) -> None:
    buffer = np.zeros(2, dtype=np.float32)
    base_freq = 100.0
    pitch = np.array([base_freq, base_freq * 2.0], dtype=np.float32)
    amp = np.array([0.2, 0.6], dtype=np.float32)
    settings = AudioExportSettings(
        sample_rate=2,
        bit_depth=32,
        output_type="cv",
        cv_base_freq=base_freq,
        cv_full_scale_volts=10.0,
    )
    path = tmp_path / "cv.wav"
    output = export_audio(
        path,
        buffer,
        settings,
        20.0,
        20000.0,
        pitch_buffer=pitch,
        amp_buffer=amp,
        amp_min=0.2,
        amp_max=0.6,
    )
    assert output.exists()
    rate, data = wavfile.read(output)
    assert rate == 2
    assert data.shape == (2, 2)
    assert np.allclose(data[:, 0], [0.0, 0.1], atol=1e-6)
    assert np.allclose(data[:, 1], [-1.0, 1.0], atol=1e-6)
