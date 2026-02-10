from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_recent_projects_path(app_name: str = "SOMA") -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support" / app_name
    elif sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) / app_name if appdata else home / "AppData" / "Roaming" / app_name
    else:
        base = home / ".config" / app_name.lower()
    return base / "recent_projects.json"


class RecentProjectStore:
    def __init__(self, path: Path, limit: int = 10) -> None:
        self._path = path
        self._limit = limit

    def list(self) -> list[dict[str, str]]:
        payload = self._load_payload()
        projects_raw = payload.get("projects")
        if not isinstance(projects_raw, list):
            return []
        projects: list[dict[str, str]] = []
        for item in projects_raw:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            last_opened_at = item.get("last_opened_at")
            if not isinstance(path, str):
                continue
            if not isinstance(last_opened_at, str):
                last_opened_at = datetime.now(UTC).isoformat()
            projects.append({"path": path, "last_opened_at": last_opened_at})
        return projects[: self._limit]

    def add(self, project_path: Path) -> None:
        normalized = str(project_path.expanduser().resolve())
        now = datetime.now(UTC).isoformat()
        current = self.list()
        deduped = [item for item in current if item["path"] != normalized]
        deduped.insert(0, {"path": normalized, "last_opened_at": now})
        self._write_payload({"projects": deduped[: self._limit]})

    def _load_payload(self) -> dict[str, Any]:
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except FileNotFoundError:
            return {"projects": []}
        except (OSError, json.JSONDecodeError):
            return {"projects": []}
        if isinstance(loaded, dict):
            return loaded
        return {"projects": []}

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(self._path)
