from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

pytest.importorskip("mido")

from soma.exporter import (  # noqa: E402
    AudioExportSettings,
    MidiExportSettings,
    MonophonicExportSettings,
    MpeExportSettings,
    MultiTrackExportSettings,
    _partial_timed_events,
    export_audio,
    export_cv_audio,
    export_monophonic_midi,
    export_mpe,
    export_multitrack_midi,
    render_cv_voice_buffers,
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


def test_export_audio_peak_normalizes_sine(tmp_path: Path) -> None:
    buffer = np.array([0.0, 2.0, -2.0], dtype=np.float32)
    settings = AudioExportSettings(sample_rate=44100, bit_depth=32, output_type="sine")
    path = tmp_path / "normalized.wav"
    output = export_audio(path, buffer, settings, 20.0, 20000.0)
    _rate, data = wavfile.read(output)
    assert np.allclose(data, [0.0, 0.99, -0.99], atol=1e-6)


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


def test_export_mpe_cc1_mapping(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=0.2),
                PartialPoint(time=0.5, freq=440.0, amp=0.4),
            ],
        )
    ]
    settings = MpeExportSettings(pitch_bend_range=48, amplitude_mapping="cc1")
    path = tmp_path / "cc1.mid"
    results = export_mpe(partials, path, settings)
    midi = pytest.importorskip("mido").MidiFile(results[0])
    cc1 = [msg.value for msg in midi.tracks[1] if msg.type == "control_change" and msg.control == 1]
    assert len(cc1) >= 2


def test_export_mpe_amp_db_curve_changes_values(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=0.2),
                PartialPoint(time=0.5, freq=440.0, amp=0.15),
                PartialPoint(time=1.0, freq=440.0, amp=0.1),
            ],
        )
    ]
    linear_settings = MpeExportSettings(pitch_bend_range=48, amplitude_mapping="pressure", amplitude_curve="linear")
    db_settings = MpeExportSettings(pitch_bend_range=48, amplitude_mapping="pressure", amplitude_curve="db")
    linear_path = export_mpe(partials, tmp_path / "linear.mid", linear_settings)[0]
    db_path = export_mpe(partials, tmp_path / "db.mid", db_settings)[0]
    linear_midi = pytest.importorskip("mido").MidiFile(linear_path)
    db_midi = pytest.importorskip("mido").MidiFile(db_path)
    linear_pressure = [msg.value for msg in linear_midi.tracks[1] if msg.type == "aftertouch"]
    db_pressure = [msg.value for msg in db_midi.tracks[1] if msg.type == "aftertouch"]
    assert len(linear_pressure) >= 3
    assert len(db_pressure) >= 3
    assert db_pressure[1] > linear_pressure[1]


def test_partial_timed_events_cc74_linear_resample_density() -> None:
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.0, freq=440.0, amp=0.0),
            PartialPoint(time=1.0, freq=440.0, amp=1.0),
        ],
    )
    settings = MidiExportSettings(amplitude_mapping="cc74", cc_update_rate_hz=4)
    events = _partial_timed_events(partial, channel=0, settings=settings, amp_min=0.0, amp_max=1.0)
    cc74 = [event for event in events if event.message.type == "control_change" and event.message.control == 74]

    assert len(cc74) == 5
    assert [event.time for event in cc74] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0], abs=1e-9)
    assert [event.message.value for event in cc74] == [1, 32, 64, 96, 127]


def test_render_cv_mono_prefers_latest_started_partial() -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=100.0, amp=0.5),
                PartialPoint(time=1.0, freq=100.0, amp=0.5),
            ],
        ),
        Partial(
            id="p2",
            points=[
                PartialPoint(time=0.5, freq=200.0, amp=0.2),
                PartialPoint(time=1.2, freq=200.0, amp=0.2),
            ],
        ),
    ]
    voice_buffers, _, _ = render_cv_voice_buffers(partials, sample_rate=10, duration_sec=1.5, mode="mono")
    pitch, amp = voice_buffers[0]
    assert pitch[2] == pytest.approx(100.0, abs=1e-6)
    assert pitch[7] == pytest.approx(200.0, abs=1e-6)
    assert amp[7] == pytest.approx(0.2, abs=1e-6)


def test_render_cv_holds_pitch_after_last_partial() -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.2, freq=300.0, amp=0.8),
                PartialPoint(time=0.4, freq=300.0, amp=0.8),
            ],
        )
    ]
    voice_buffers, _, _ = render_cv_voice_buffers(partials, sample_rate=10, duration_sec=1.0, mode="mono")
    pitch, amp = voice_buffers[0]
    assert pitch[1] == pytest.approx(0.0, abs=1e-6)
    assert pitch[5] == pytest.approx(300.0, abs=1e-6)
    assert amp[5] == pytest.approx(0.0, abs=1e-6)


def test_export_cv_audio_poly_outputs_per_voice_files(tmp_path: Path) -> None:
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=110.0, amp=0.3),
                PartialPoint(time=1.0, freq=110.0, amp=0.3),
            ],
        ),
        Partial(
            id="p2",
            points=[
                PartialPoint(time=0.2, freq=220.0, amp=0.5),
                PartialPoint(time=1.2, freq=220.0, amp=0.5),
            ],
        ),
    ]
    settings = AudioExportSettings(sample_rate=10, bit_depth=32, output_type="cv", cv_mode="poly")
    voice_buffers, amp_min, amp_max = render_cv_voice_buffers(partials, sample_rate=10, duration_sec=1.5, mode="poly")
    output = export_cv_audio(tmp_path / "poly.wav", settings, voice_buffers, amp_min, amp_max)
    assert len(output) == 2
    assert output[0].name.endswith("_01.wav")
    assert output[1].name.endswith("_02.wav")
    assert output[0].exists()
    assert output[1].exists()
