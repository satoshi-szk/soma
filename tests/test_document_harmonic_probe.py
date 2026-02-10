import numpy as np

from soma.document import SomaDocument, _partial_sample_at_time
from soma.models import AudioInfo, Partial, PartialPoint


def _make_doc(sample_rate: int = 48000, duration_sec: float = 1.0) -> SomaDocument:
    doc = SomaDocument()
    doc.audio_info = AudioInfo(
        path="/tmp/test.wav",
        name="test.wav",
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        channels=1,
        truncated=False,
    )
    doc.audio_data = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    doc.synth.reset(sample_rate=sample_rate, duration_sec=duration_sec)
    return doc


def test_partial_sample_at_time_interpolates_freq_and_amp() -> None:
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.0, freq=100.0, amp=0.2),
            PartialPoint(time=1.0, freq=200.0, amp=0.6),
        ],
    )

    sample = _partial_sample_at_time(partial, 0.25)

    assert sample is not None
    freq, amp = sample
    assert freq == 125.0
    assert amp == 0.3


def test_harmonic_probe_tones_returns_active_partials_at_playhead() -> None:
    doc = _make_doc()
    active = Partial(
        id="active",
        points=[
            PartialPoint(time=0.0, freq=440.0, amp=0.5),
            PartialPoint(time=1.0, freq=660.0, amp=0.7),
        ],
    )
    muted = Partial(
        id="muted",
        points=[
            PartialPoint(time=0.0, freq=330.0, amp=0.4),
            PartialPoint(time=1.0, freq=330.0, amp=0.4),
        ],
        is_muted=True,
    )
    outside = Partial(
        id="outside",
        points=[
            PartialPoint(time=1.1, freq=500.0, amp=0.9),
            PartialPoint(time=1.5, freq=500.0, amp=0.9),
        ],
    )
    doc.store.add(active)
    doc.store.add(muted)
    doc.store.add(outside)

    freqs, amps = doc._harmonic_probe_tones(0.5)

    assert freqs.tolist() == [550.0]
    assert amps.tolist() == [0.6]


def test_update_harmonic_probe_allows_silent_position() -> None:
    doc = _make_doc()
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.0, freq=440.0, amp=0.5),
            PartialPoint(time=0.5, freq=440.0, amp=0.5),
        ],
    )
    doc.store.add(partial)
    doc._playback_mode = "probe"
    captured: dict[str, list[float]] = {}

    def fake_update_probe(freqs: np.ndarray, amps: np.ndarray) -> bool:
        captured["freqs"] = freqs.tolist()
        captured["amps"] = amps.tolist()
        return True

    doc.player.update_probe = fake_update_probe  # type: ignore[method-assign]
    ok = doc.update_harmonic_probe(0.8)

    assert ok is True
    assert captured["freqs"] == []
    assert captured["amps"] == []


def test_start_harmonic_probe_allows_silent_position() -> None:
    doc = _make_doc()
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.2, freq=440.0, amp=0.5),
            PartialPoint(time=0.4, freq=440.0, amp=0.5),
        ],
    )
    doc.store.add(partial)
    captured: dict[str, list[float]] = {}

    def fake_play_probe(freqs: np.ndarray, amps: np.ndarray) -> bool:
        captured["freqs"] = freqs.tolist()
        captured["amps"] = amps.tolist()
        return True

    doc.player.play_probe = fake_play_probe  # type: ignore[method-assign]
    ok = doc.start_harmonic_probe(0.8)

    assert ok is True
    assert doc._playback_mode == "probe"
    assert captured["freqs"] == []
    assert captured["amps"] == []


def test_start_harmonic_probe_uses_midi_output_when_playback_mode_is_midi() -> None:
    doc = _make_doc()
    doc._playback_output_mode = "midi"
    doc._midi_output_name = "Dummy MIDI Output"
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.0, freq=440.0, amp=0.5),
            PartialPoint(time=1.0, freq=440.0, amp=0.5),
        ],
    )
    doc.store.add(partial)
    captured: dict[str, object] = {}

    def fake_start_probe(
        output_name: str,
        partial_ids: list[str],
        freqs: list[float],
        amps: list[float],
        pitch_bend_range: int,
    ) -> bool:
        captured["output_name"] = output_name
        captured["partial_ids"] = partial_ids
        captured["freqs"] = freqs
        captured["amps"] = amps
        captured["pitch_bend_range"] = pitch_bend_range
        return True

    doc.midi_player.start_probe = fake_start_probe  # type: ignore[method-assign]
    ok = doc.start_harmonic_probe(0.5)

    assert ok is True
    assert doc._playback_mode == "probe"
    assert captured["output_name"] == "Dummy MIDI Output"
    assert captured["partial_ids"] == ["p1"]
    assert captured["freqs"] == [440.0]
    assert captured["amps"] == [0.5]
    assert captured["pitch_bend_range"] == 48


def test_update_harmonic_probe_uses_midi_output_when_playback_mode_is_midi() -> None:
    doc = _make_doc()
    doc._playback_mode = "probe"
    doc._playback_output_mode = "midi"
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.0, freq=330.0, amp=0.4),
            PartialPoint(time=1.0, freq=330.0, amp=0.6),
        ],
    )
    doc.store.add(partial)
    captured: dict[str, list[float]] = {}

    def fake_update_probe(partial_ids: list[str], freqs: list[float], amps: list[float]) -> bool:
        captured["partial_ids"] = partial_ids
        captured["freqs"] = freqs
        captured["amps"] = amps
        return True

    doc.midi_player.update_probe = fake_update_probe  # type: ignore[method-assign]
    ok = doc.update_harmonic_probe(0.5)

    assert ok is True
    assert captured["partial_ids"] == ["p1"]
    assert captured["freqs"] == [330.0]
    assert captured["amps"] == [0.5]
