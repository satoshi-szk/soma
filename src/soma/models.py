from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class AudioInfo:
    path: str
    name: str
    sample_rate: int
    duration_sec: float
    channels: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectrogramPreview:
    width: int
    height: int
    data: list[int]
    freq_min: float
    freq_max: float
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisSettings:
    freq_min: float = 20.0
    freq_max: float = 20000.0
    bins_per_octave: int = 48
    time_resolution_ms: float = 10.0
    preview_freq_max: float = 12000.0
    preview_bins_per_octave: int = 12
    color_map: str = "magma"
    brightness: float = 0.0
    contrast: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartialPoint:
    time: float
    freq: float
    amp: float

    def to_list(self) -> list[float]:
        return [self.time, self.freq, self.amp]


@dataclass
class Partial:
    id: str
    points: list[PartialPoint]
    is_muted: bool = False

    def sorted_points(self) -> list[PartialPoint]:
        return sorted(self.points, key=lambda p: p.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "is_muted": self.is_muted,
            "points": [point.to_list() for point in self.sorted_points()],
        }

    @classmethod
    def from_points(cls, partial_id: str, points: Iterable[PartialPoint]) -> Partial:
        return cls(id=partial_id, points=list(points))


@dataclass(frozen=True)
class ProjectMeta:
    format_version: str = "1.0.0"
    app_name: str = "SOMA"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceInfo:
    file_path: str
    sample_rate: int
    duration_sec: float
    md5_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
