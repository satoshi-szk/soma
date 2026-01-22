from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from soma.models import AnalysisSettings, Partial, PartialPoint, ProjectMeta, SourceInfo


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_project_payload(
    source: SourceInfo,
    settings: AnalysisSettings,
    partials: list[Partial],
    created_at: str | None = None,
) -> dict[str, Any]:
    meta = ProjectMeta(created_at=created_at or datetime.now(UTC).isoformat())
    return {
        "meta": meta.to_dict(),
        "source": source.to_dict(),
        "analysis_settings": settings.to_dict(),
        "data": {"partials": [partial.to_dict() for partial in partials]},
    }


def save_project(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_project(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return cast(dict[str, Any], data)


def parse_settings(data: dict[str, Any]) -> AnalysisSettings:
    settings = data.get("analysis_settings", {})
    return AnalysisSettings(
        freq_min=float(settings.get("freq_min", 20.0)),
        freq_max=float(settings.get("freq_max", 20000.0)),
        bins_per_octave=int(settings.get("bins_per_octave", 48)),
        time_resolution_ms=float(settings.get("time_resolution_ms", 10.0)),
        preview_freq_max=float(settings.get("preview_freq_max", 12000.0)),
        preview_bins_per_octave=int(settings.get("preview_bins_per_octave", 12)),
        color_map=str(settings.get("color_map", "magma")),
        brightness=float(settings.get("brightness", 0.0)),
        contrast=float(settings.get("contrast", 1.0)),
    )


def parse_source(data: dict[str, Any]) -> SourceInfo | None:
    source = data.get("source")
    if not isinstance(source, dict):
        return None
    return SourceInfo(
        file_path=str(source.get("file_path", "")),
        sample_rate=int(source.get("sample_rate", 0)),
        duration_sec=float(source.get("duration_sec", 0.0)),
        md5_hash=str(source.get("md5_hash", "")),
    )


def parse_partials(data: dict[str, Any]) -> list[Partial]:
    partials_data = data.get("data", {}).get("partials", [])
    partials: list[Partial] = []
    if not isinstance(partials_data, list):
        return partials
    for entry in partials_data:
        if not isinstance(entry, dict):
            continue
        points = []
        for raw in entry.get("points", []):
            if not isinstance(raw, (list, tuple)) or len(raw) < 3:
                continue
            points.append(PartialPoint(time=float(raw[0]), freq=float(raw[1]), amp=float(raw[2])))
        partials.append(
            Partial(
                id=str(entry.get("id", "")),
                points=points,
                is_muted=bool(entry.get("is_muted", False)),
            )
        )
    return partials
