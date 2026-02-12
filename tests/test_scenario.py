from soma.app import SomaApi
from soma.models import Partial, PartialPoint


def _partial(partial_id: str, times: list[float], freq: float = 440.0) -> Partial:
    points = [PartialPoint(time=t, freq=freq, amp=0.5) for t in times]
    return Partial(id=partial_id, points=points)


def test_api_undo_redo_merge_cycle() -> None:
    api = SomaApi()
    session = api._session
    session.synth.reset(sample_rate=44100, duration_sec=1.0)

    first = _partial("p1", [0.0, 0.05, 0.1])
    second = _partial("p2", [0.2, 0.25, 0.3])
    session.store.add(first)
    session.store.add(second)
    session.synth.apply_partial(first)
    session.synth.apply_partial(second)

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


def test_api_update_delete_undo_redo_cycle() -> None:
    api = SomaApi()
    session = api._session
    session.synth.reset(sample_rate=44100, duration_sec=1.0)

    # セットアップ: partial を作成
    original = _partial("p1", [0.0, 0.05, 0.1], freq=440.0)
    session.store.add(original)
    session.synth.apply_partial(original)

    # 1. update_partial でポイントを編集 (周波数を変更)
    updated_points = [[0.0, 500.0, 0.5], [0.05, 510.0, 0.5], [0.1, 520.0, 0.5]]
    update_result = api.update_partial({"id": "p1", "points": updated_points})
    assert update_result["status"] == "ok"
    assert update_result["partial"]["points"][0][1] == 500.0  # freq が更新された

    # 2. delete_partials で削除
    delete_result = api.delete_partials({"ids": ["p1"]})
    assert delete_result["status"] == "ok"
    assert session.store.get("p1") is None

    # 3. undo → delete 取り消し (partial 復活、update 後の状態)
    undo1 = api.undo()
    assert undo1["status"] == "ok"
    restored = next((p for p in undo1["partials"] if p["id"] == "p1"), None)
    assert restored is not None
    assert restored["points"][0][1] == 500.0  # update 後の状態で復活

    # 4. undo → update 取り消し (元の状態)
    undo2 = api.undo()
    assert undo2["status"] == "ok"
    original_restored = next((p for p in undo2["partials"] if p["id"] == "p1"), None)
    assert original_restored is not None
    assert original_restored["points"][0][1] == 440.0  # 元の freq に戻る

    # 5. redo → update 再適用
    redo1 = api.redo()
    assert redo1["status"] == "ok"
    updated_again = next((p for p in redo1["partials"] if p["id"] == "p1"), None)
    assert updated_again is not None
    assert updated_again["points"][0][1] == 500.0

    # 6. redo → delete 再適用
    redo2 = api.redo()
    assert redo2["status"] == "ok"
    assert all(p["id"] != "p1" for p in redo2["partials"])
