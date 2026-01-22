from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpdateSettingsPayload(PayloadBase):
    freq_min: float = 20.0
    freq_max: float = 20000.0
    bins_per_octave: int = 48
    time_resolution_ms: float = 10.0
    preview_freq_max: float = 12000.0
    preview_bins_per_octave: int = 12
    wavelet_bandwidth: float = 8.0
    wavelet_center_freq: float = 2.0
    color_map: str = "magma"
    brightness: float = 0.0
    contrast: float = 1.0


class TracePartialPayload(PayloadBase):
    trace: list[tuple[float, float]] = Field(default_factory=list)


class ErasePartialPayload(PayloadBase):
    trace: list[tuple[float, float]] = Field(default_factory=list)
    radius_hz: float = 40.0


class UpdatePartialPayload(PayloadBase):
    id: str
    points: list[tuple[float, float, float]]


class MergePartialsPayload(PayloadBase):
    first: str
    second: str


class DeletePartialsPayload(PayloadBase):
    ids: list[str] = Field(default_factory=list)


class ToggleMutePayload(PayloadBase):
    id: str


class HitTestPayload(PayloadBase):
    time: float
    freq: float
    tolerance: float = 0.05


class SelectInBoxPayload(PayloadBase):
    time_start: float
    time_end: float
    freq_start: float
    freq_end: float


class PlayPayload(PayloadBase):
    mix_ratio: float = 0.5
    loop: bool = False


class ExportMpePayload(PayloadBase):
    pitch_bend_range: int = 48
    amplitude_mapping: str = "velocity"
    bpm: float = 120.0


class ExportAudioPayload(PayloadBase):
    sample_rate: int | None = None
    bit_depth: int = 16
    output_type: str = "sine"


class RequestViewportPreviewPayload(PayloadBase):
    time_start: float
    time_end: float
    freq_min: float
    freq_max: float
    width: int
    height: int


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for item in exc.errors():
        loc = ".".join(str(piece) for piece in item.get("loc", [])) or "payload"
        msg = item.get("msg", "invalid value")
        parts.append(f"{loc}: {msg}")
    details = "; ".join(parts) if parts else "invalid payload"
    return f"Invalid payload: {details}"


def parse_payload(model: type[PayloadBase], payload: Any) -> tuple[PayloadBase | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Invalid payload: expected object."
    try:
        return model.model_validate(payload), None
    except ValidationError as exc:
        return None, _format_validation_error(exc)
