import numpy as np

from soma.document import SomaDocument
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


def test_mix_buffer_peak_normalize_prevents_clipping() -> None:
    doc = _make_doc()
    partials = [
        Partial(
            id="p1",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=1.0),
                PartialPoint(time=1.0, freq=440.0, amp=1.0),
            ],
        ),
        Partial(
            id="p2",
            points=[
                PartialPoint(time=0.0, freq=440.0, amp=1.0),
                PartialPoint(time=1.0, freq=440.0, amp=1.0),
            ],
        ),
    ]
    for partial in partials:
        doc.synth.apply_partial(partial)

    mixed = doc._mix_buffer(1.0)
    peak = float(np.max(np.abs(mixed)))

    assert peak <= 0.99 + 1e-6
    assert np.sum(np.abs(mixed) > 1.0) == 0


def test_mix_buffer_peak_normalize_silence_keeps_silence() -> None:
    doc = _make_doc()
    mixed = doc._mix_buffer(1.0)
    assert np.allclose(mixed, 0.0)


def test_playback_position_and_start_are_mapped_by_speed_ratio() -> None:
    doc = _make_doc()
    captured: dict[str, float] = {}

    def fake_play(*, loop: bool, start_position_sec: float = 0.0) -> None:
        del loop
        captured["start_position_sec"] = start_position_sec

    doc.player.play = fake_play  # type: ignore[method-assign]
    doc.player.is_playing = lambda: False  # type: ignore[method-assign]
    doc.player.position_sec = lambda: 3.0  # type: ignore[method-assign]

    doc.play(mix_ratio=0.5, loop=False, start_position_sec=2.0, speed_ratio=0.5)

    assert captured["start_position_sec"] == 4.0
    assert doc.playback_position() == 1.5
