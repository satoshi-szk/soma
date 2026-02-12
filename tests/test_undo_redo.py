from soma.models import Partial, PartialPoint
from soma.services.history import HistoryService
from soma.services.partial_edit_service import PartialEditService
from soma.services.playback_service import PlaybackService
from soma.session import ProjectSession


def _partial(partial_id: str, times: list[float]) -> Partial:
    points = [PartialPoint(time=t, freq=440.0, amp=0.5) for t in times]
    return Partial(id=partial_id, points=points)


def _stack() -> tuple[ProjectSession, HistoryService, PartialEditService, PlaybackService]:
    session = ProjectSession()
    playback = PlaybackService(session)
    history = HistoryService(session)
    history.set_callbacks(
        on_settings_applied=lambda: None,
        on_partials_changed=playback.invalidate_cache,
    )
    partial_edit = PartialEditService(session, history, on_partials_changed=playback.invalidate_cache)
    return session, history, partial_edit, playback


def test_undo_redo_restores_erased_segments() -> None:
    session, history, partial_edit, _playback = _stack()
    session.synth.reset(sample_rate=44100, duration_sec=1.0)
    original = _partial("p1", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    session.store.add(original)
    session.synth.apply_partial(original)

    trace = [(0.1, 440.0), (0.16, 440.0)]
    partial_edit.erase_path(trace, radius_hz=40.0)

    assert session.store.get("p1") is None
    assert len(session.store.all()) == 2
    segment_ids = {partial.id for partial in session.store.all()}

    history.undo()

    assert session.store.get("p1") is not None
    assert len(session.store.all()) == 1

    history.redo()

    assert session.store.get("p1") is None
    assert {partial.id for partial in session.store.all()} == segment_ids


def test_erase_path_respects_radius_hz_on_split() -> None:
    session, _history, partial_edit, _playback = _stack()
    session.synth.reset(sample_rate=44100, duration_sec=1.0)
    original = Partial(
        id="p1",
        points=[
            PartialPoint(time=0.00, freq=440.0, amp=0.5),
            PartialPoint(time=0.05, freq=440.0, amp=0.5),
            PartialPoint(time=0.10, freq=440.0, amp=0.5),
            PartialPoint(time=0.12, freq=900.0, amp=0.5),
            PartialPoint(time=0.15, freq=440.0, amp=0.5),
            PartialPoint(time=0.20, freq=900.0, amp=0.5),
        ],
    )
    session.store.add(original)
    session.synth.apply_partial(original)

    trace = [(0.10, 440.0), (0.16, 440.0)]
    partial_edit.erase_path(trace, radius_hz=60.0)

    remaining_points = [point for partial in session.store.all() for point in partial.points]
    assert all(not (point.time == 0.10 and point.freq == 440.0) for point in remaining_points)
    assert all(not (point.time == 0.15 and point.freq == 440.0) for point in remaining_points)
    assert any(point.time == 0.12 and point.freq == 900.0 for point in remaining_points)
