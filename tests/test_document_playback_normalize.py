import threading

import numpy as np

import soma.services.playback_service as playback_module
from soma.models import AudioInfo, Partial, PartialPoint
from soma.services.playback_service import PlaybackService
from soma.session import ProjectSession


def _make_session_and_playback(
    sample_rate: int = 48000,
    duration_sec: float = 1.0,
) -> tuple[ProjectSession, PlaybackService]:
    session = ProjectSession()
    session.audio_info = AudioInfo(
        path="/tmp/test.wav",
        name="test.wav",
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        channels=1,
        truncated=False,
    )
    session.audio_data = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
    session.synth.reset(sample_rate=sample_rate, duration_sec=duration_sec)
    playback = PlaybackService(session)
    return session, playback


def test_mix_buffer_peak_normalize_prevents_clipping() -> None:
    session, playback = _make_session_and_playback()
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
        session.synth.apply_partial(partial)

    mixed = playback._mix_buffer(1.0)
    peak = float(np.max(np.abs(mixed)))

    assert peak <= 0.99 + 1e-6
    assert np.sum(np.abs(mixed) > 1.0) == 0


def test_mix_buffer_peak_normalize_silence_keeps_silence() -> None:
    _session, playback = _make_session_and_playback()
    mixed = playback._mix_buffer(1.0)
    assert np.allclose(mixed, 0.0)


def test_playback_position_and_start_are_mapped_by_speed_ratio() -> None:
    _session, playback = _make_session_and_playback()
    captured: dict[str, float] = {}
    started = threading.Event()

    def fake_play(*, loop: bool, start_position_sec: float = 0.0) -> None:
        del loop
        captured["start_position_sec"] = start_position_sec
        started.set()

    playback._session.player.play = fake_play  # type: ignore[method-assign]
    playback._session.player.is_playing = lambda: False  # type: ignore[method-assign]
    playback._session.player.position_sec = lambda: 3.0  # type: ignore[method-assign]

    playback.play(mix_ratio=0.5, loop=False, start_position_sec=2.0, speed_ratio=0.5)

    assert started.wait(timeout=1.0)
    assert captured["start_position_sec"] == 4.0
    assert playback.playback_position() == 1.5


def test_is_playing_excludes_prepare_state() -> None:
    _session, playback = _make_session_and_playback()
    playback._session._playback_mode = "normal"
    playback._session._is_preparing_playback = True
    playback._session.player.is_playing = lambda: False  # type: ignore[method-assign]

    assert playback.is_preparing_playback() is True
    assert playback.is_playing() is False


def test_playback_uses_cached_stretched_buffer(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _session, playback = _make_session_and_playback(duration_sec=2.0)
    calls = {"stretch": 0, "play": 0}
    started = threading.Event()

    def fake_stretch(buffer: np.ndarray, speed_ratio: float, sample_rate: int, mode: str = "librosa") -> np.ndarray:
        del speed_ratio, sample_rate, mode
        calls["stretch"] += 1
        return buffer

    def fake_play(*, loop: bool, start_position_sec: float = 0.0) -> None:
        del loop, start_position_sec
        calls["play"] += 1
        started.set()

    monkeypatch.setattr(playback_module, "time_stretch_pitch_preserving", fake_stretch)
    playback._session.player.play = fake_play  # type: ignore[method-assign]
    playback._session.player.is_playing = lambda: False  # type: ignore[method-assign]

    playback.play(mix_ratio=0.55, loop=False, start_position_sec=0.5, speed_ratio=2.0, time_stretch_mode="native")
    assert started.wait(timeout=1.0)

    started.clear()
    playback.play(mix_ratio=0.55, loop=False, start_position_sec=1.0, speed_ratio=2.0, time_stretch_mode="native")
    assert started.wait(timeout=1.0)

    assert calls["play"] == 2
    assert calls["stretch"] == 1


def test_playback_resynth_only_skips_time_stretch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    session, playback = _make_session_and_playback(duration_sec=2.0)
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.0, freq=440.0, amp=0.5),
            PartialPoint(time=2.0, freq=440.0, amp=0.5),
        ],
    )
    session.store.add(partial)
    session.synth.apply_partial(partial)
    started = threading.Event()

    def should_not_be_called(*_args, **_kwargs) -> np.ndarray:  # type: ignore[no-untyped-def]
        raise AssertionError("time stretch should be skipped for resynth-only playback")

    def fake_play(*, loop: bool, start_position_sec: float = 0.0) -> None:
        del loop, start_position_sec
        started.set()

    monkeypatch.setattr(playback_module, "time_stretch_pitch_preserving", should_not_be_called)
    playback._session.player.play = fake_play  # type: ignore[method-assign]
    playback._session.player.is_playing = lambda: False  # type: ignore[method-assign]

    playback.play(mix_ratio=1.0, loop=False, start_position_sec=0.0, speed_ratio=2.0, time_stretch_mode="librosa")

    assert started.wait(timeout=1.0)


def test_update_mix_ratio_does_not_seek_backwards(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _session, playback = _make_session_and_playback(duration_sec=2.0)
    playback._session._playback_mode = "normal"
    playback._session._playback_speed_ratio = 2.0
    playback._session.player.is_playing = lambda: True  # type: ignore[method-assign]

    monkeypatch.setattr(
        playback,
        "_build_speed_adjusted_buffer",
        lambda *_args, **_kwargs: np.zeros(32, dtype=np.float32),
    )

    captured: dict[str, object] = {}

    def fake_update_buffer(buffer: np.ndarray, start_position_sec: float | None = None) -> None:
        captured["size"] = buffer.size
        captured["start_position_sec"] = start_position_sec

    playback._session.player.update_buffer = fake_update_buffer  # type: ignore[method-assign]

    updated = playback.update_mix_ratio(0.7)

    assert updated is True
    assert captured["size"] == 32
    assert captured["start_position_sec"] is None


def test_update_mix_ratio_reuses_stretched_original_cache(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    session, playback = _make_session_and_playback(duration_sec=2.0)
    partial = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.0, freq=440.0, amp=0.5),
            PartialPoint(time=2.0, freq=440.0, amp=0.5),
        ],
    )
    session.store.add(partial)
    session.synth.apply_partial(partial)
    playback._session._playback_mode = "normal"
    playback._session._playback_speed_ratio = 2.0
    playback._session._last_time_stretch_mode = "native"
    playback._session.player.is_playing = lambda: True  # type: ignore[method-assign]
    playback._session.player.update_buffer = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    calls = {"stretch": 0}

    def fake_stretch(buffer: np.ndarray, speed_ratio: float, sample_rate: int, mode: str = "librosa") -> np.ndarray:
        del speed_ratio, sample_rate, mode
        calls["stretch"] += 1
        return buffer

    monkeypatch.setattr(playback_module, "time_stretch_pitch_preserving", fake_stretch)

    updated_first = playback.update_mix_ratio(0.4)
    updated_second = playback.update_mix_ratio(0.7)

    assert updated_first is True
    assert updated_second is True
    assert calls["stretch"] == 1
