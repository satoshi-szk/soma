from pathlib import Path

from soma.models import (
    AnalysisSettings,
    Partial,
    PartialPoint,
    PlaybackSettings,
    SnapSettings,
    SourceInfo,
    SpectrogramSettings,
)
from soma.persistence import (
    build_project_payload,
    load_project,
    parse_partials,
    parse_playback_settings,
    parse_settings,
    parse_source,
    save_project,
)


def test_save_and_load_project(tmp_path: Path) -> None:
    path = tmp_path / "project.soma"
    source = SourceInfo(
        file_path="/tmp/audio.wav",
        sample_rate=44100,
        duration_sec=1.0,
        md5_hash="abc",
    )
    settings = AnalysisSettings(
        spectrogram=SpectrogramSettings(freq_min=30.0, freq_max=18000.0),
        snap=SnapSettings(freq_min=30.0, freq_max=18000.0),
    )
    playback_settings = PlaybackSettings(
        master_volume=0.35,
        output_mode="midi",
        mix_ratio=0.42,
        speed_ratio=2.0,
        time_stretch_mode="native",
        midi_mode="multitrack",
        midi_output_name="IAC Driver Bus 1",
        midi_pitch_bend_range=24,
        midi_amplitude_mapping="cc1",
        midi_amplitude_curve="db",
        midi_bpm=96.0,
    )
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.1, freq=440.0, amp=0.5),
                PartialPoint(time=0.2, freq=441.0, amp=0.5),
            ],
        )
    ]
    payload = build_project_payload(source, settings, playback_settings, partials, created_at="2024-01-01T00:00:00Z")
    save_project(path, payload)

    loaded = load_project(path)
    loaded_settings = parse_settings(loaded)
    loaded_source = parse_source(loaded)
    loaded_playback_settings = parse_playback_settings(loaded)
    loaded_partials = parse_partials(loaded)

    assert loaded_source is not None
    assert loaded_source.file_path == source.file_path
    assert loaded_settings.spectrogram.freq_min == 30.0
    assert loaded_playback_settings.master_volume == 0.35
    assert loaded_playback_settings.output_mode == "midi"
    assert loaded_playback_settings.mix_ratio == 0.42
    assert loaded_playback_settings.speed_ratio == 2.0
    assert loaded_playback_settings.time_stretch_mode == "native"
    assert loaded_playback_settings.midi_mode == "multitrack"
    assert loaded_playback_settings.midi_output_name == "IAC Driver Bus 1"
    assert loaded_playback_settings.midi_pitch_bend_range == 24
    assert loaded_playback_settings.midi_amplitude_mapping == "cc1"
    assert loaded_playback_settings.midi_amplitude_curve == "db"
    assert loaded_playback_settings.midi_bpm == 96.0
    assert len(loaded_partials) == 1
    assert loaded_partials[0].id == "p1"
