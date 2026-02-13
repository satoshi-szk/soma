from __future__ import annotations

import colorsys
import random
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
    data: list[int] | bytes
    time_start: float
    time_end: float
    freq_min: float
    freq_max: float
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if isinstance(self.data, bytes):
            # JSON 応答時のみ list へ変換する。通常のキャッシュ経路では bytes のまま使う。
            payload["data"] = list(self.data)
        return payload


@dataclass(frozen=True)
class SpectrogramSettings:
    freq_min: float = 20.0
    freq_max: float = 20000.0
    preview_freq_max: float = 12000.0
    multires_blend_octaves: float = 1.0
    multires_window_size_scale: float = 1.0
    gain: float = 1.0
    min_db: float = -80.0
    max_db: float = 0.0
    gamma: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapSettings:
    freq_min: float = 20.0
    freq_max: float = 20000.0
    bins_per_octave: int = 96
    time_resolution_ms: float = 10.0
    wavelet_bandwidth: float = 8.0
    wavelet_center_freq: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisSettings:
    spectrogram: SpectrogramSettings = field(default_factory=SpectrogramSettings)
    snap: SnapSettings = field(default_factory=SnapSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spectrogram": self.spectrogram.to_dict(),
            "snap": self.snap.to_dict(),
        }


@dataclass(frozen=True)
class PlaybackSettings:
    master_volume: float = 1.0
    output_mode: str = "audio"  # audio | midi
    mix_ratio: float = 0.55
    speed_ratio: float = 1.0
    time_stretch_mode: str = "librosa"  # native | librosa
    midi_mode: str = "mpe"  # mpe | multitrack | mono
    midi_output_name: str = ""
    midi_pitch_bend_range: int = 48
    midi_amplitude_mapping: str = "cc74"
    midi_amplitude_curve: str = "linear"
    midi_cc_update_rate_hz: int = 400
    midi_bpm: float = 120.0

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
    color: str = field(default_factory=lambda: generate_bright_color())

    def sorted_points(self) -> list[PartialPoint]:
        return sorted(self.points, key=lambda p: p.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "is_muted": self.is_muted,
            "color": self.color,
            "points": [point.to_list() for point in self.sorted_points()],
        }

    @classmethod
    def from_points(cls, partial_id: str, points: Iterable[PartialPoint], color: str | None = None) -> Partial:
        return cls(id=partial_id, points=list(points), color=color or generate_bright_color())


def generate_bright_color() -> str:
    hue = random.random()
    saturation = random.uniform(0.6, 0.95)
    value = random.uniform(0.85, 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


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
