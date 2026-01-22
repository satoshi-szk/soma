from soma.document import SomaDocument
from soma.models import Partial, PartialPoint


def _partial(partial_id: str, times: list[float]) -> Partial:
    points = [PartialPoint(time=t, freq=440.0, amp=0.5) for t in times]
    return Partial(id=partial_id, points=points)


def test_undo_redo_restores_erased_segments() -> None:
    doc = SomaDocument()
    doc.synth.reset(sample_rate=44100, duration_sec=1.0)
    original = _partial("p1", [0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    doc.store.add(original)
    doc.synth.apply_partial(original)

    trace = [(0.1, 440.0), (0.16, 440.0)]
    doc.erase_path(trace, radius_hz=40.0)

    assert doc.store.get("p1") is None
    assert len(doc.store.all()) == 2
    segment_ids = {partial.id for partial in doc.store.all()}

    doc.undo()

    assert doc.store.get("p1") is not None
    assert len(doc.store.all()) == 1

    doc.redo()

    assert doc.store.get("p1") is None
    assert {partial.id for partial in doc.store.all()} == segment_ids


