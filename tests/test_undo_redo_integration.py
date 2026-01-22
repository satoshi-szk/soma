from soma.app import SomaApi
from soma.models import Partial, PartialPoint


def _partial(partial_id: str, times: list[float]) -> Partial:
    points = [PartialPoint(time=t, freq=440.0, amp=0.5) for t in times]
    return Partial(id=partial_id, points=points)


def test_api_undo_redo_merge_cycle() -> None:
    api = SomaApi()
    doc = api._doc
    doc.synth.reset(sample_rate=44100, duration_sec=1.0)

    first = _partial("p1", [0.0, 0.05, 0.1])
    second = _partial("p2", [0.2, 0.25, 0.3])
    doc.store.add(first)
    doc.store.add(second)
    doc.synth.apply_partial(first)
    doc.synth.apply_partial(second)

    merge_result = api.merge_partials({"first": "p1", "second": "p2"})
    assert merge_result["status"] == "ok"
    merged_id = merge_result["partial"]["id"]

    undo_result = api.undo()
    assert undo_result["status"] == "ok"
    undo_ids = {partial["id"] for partial in undo_result["partials"]}
    assert undo_ids == {"p1", "p2"}

    redo_result = api.redo()
    assert redo_result["status"] == "ok"
    redo_ids = {partial["id"] for partial in redo_result["partials"]}
    assert redo_ids == {merged_id}
