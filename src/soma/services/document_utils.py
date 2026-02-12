from __future__ import annotations

import uuid
from pathlib import Path

from soma.models import Partial, PartialPoint, generate_bright_color


def ensure_soma_extension(path: Path) -> Path:
    if path.name.lower().endswith(".soma"):
        return path
    return path.with_name(f"{path.name}.soma")


def sanitize_audio_filename(name: str, fallback: str) -> str:
    candidate = Path(name).name.strip()
    if not candidate or candidate in {".", ".."}:
        candidate = Path(fallback).name
    return candidate.replace("/", "_").replace("\\", "_")


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def partial_sample_at_time(partial: Partial, time_sec: float) -> tuple[float, float] | None:
    points = partial.sorted_points()
    if len(points) < 2:
        return None
    if time_sec < points[0].time or time_sec > points[-1].time:
        return None
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        if end.time < start.time:
            continue
        if time_sec < start.time or time_sec > end.time:
            continue
        dt = end.time - start.time
        if dt <= 1e-9:
            return end.freq, end.amp
        ratio = (time_sec - start.time) / dt
        freq = start.freq + (end.freq - start.freq) * ratio
        amp = start.amp + (end.amp - start.amp) * ratio
        return float(freq), float(amp)
    return float(points[-1].freq), float(points[-1].amp)


def intersects_trace(partial: Partial, trace: list[tuple[float, float]], radius_hz: float) -> bool:
    return any(point_in_erase_path(point, trace, radius_hz) for point in partial.points)


def point_in_erase_path(point: PartialPoint, trace: list[tuple[float, float]], radius_hz: float) -> bool:
    for time_sec, freq_hz in trace:
        if abs(point.time - time_sec) <= 0.02 and abs(point.freq - freq_hz) <= radius_hz:
            return True
    return False


def split_partial(
    partial: Partial,
    trace: list[tuple[float, float]],
    radius_hz: float,
) -> list[Partial]:
    remaining = [p for p in partial.points if not point_in_erase_path(p, trace, radius_hz)]
    if not remaining:
        return []
    remaining.sort(key=lambda p: p.time)
    segments: list[list[PartialPoint]] = [[]]
    for point in remaining:
        if not segments[-1]:
            segments[-1].append(point)
            continue
        if point.time - segments[-1][-1].time > 0.1:
            segments.append([point])
        else:
            segments[-1].append(point)
    result: list[Partial] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        result.append(Partial(id=str(uuid.uuid4()), points=segment, color=generate_bright_color()))
    return result
