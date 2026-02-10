from pathlib import Path

from soma.recent_projects import RecentProjectStore


def test_recent_projects_store_dedupes_and_orders(tmp_path: Path) -> None:
    store = RecentProjectStore(tmp_path / "recent_projects.json", limit=3)
    a = tmp_path / "a.soma"
    b = tmp_path / "b.soma"
    c = tmp_path / "c.soma"
    d = tmp_path / "d.soma"

    store.add(a)
    store.add(b)
    store.add(c)
    store.add(b)
    store.add(d)

    rows = store.list()
    assert [row["path"] for row in rows] == [str(d.resolve()), str(b.resolve()), str(c.resolve())]


def test_recent_projects_store_handles_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "recent_projects.json"
    path.write_text("{invalid json", encoding="utf-8")
    store = RecentProjectStore(path, limit=10)
    assert store.list() == []
