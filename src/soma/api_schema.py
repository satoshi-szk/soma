from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class PayloadBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpectrogramSettingsPayload(PayloadBase):
    method: str = Field(default="multires_stft", pattern="^(multires_stft|reassigned_stft)$")
    freq_min: float = 20.0
    freq_max: float = 20000.0
    preview_freq_max: float = 12000.0
    multires_blend_octaves: float = 1.0
    multires_window_size_scale: float = 1.0
    reassigned_ref_power: float = Field(default=1e-6, ge=0.0)
    gain: float = 1.0
    min_db: float = -80.0
    max_db: float = 0.0
    gamma: float = 1.0


class SnapSettingsPayload(PayloadBase):
    freq_min: float = 20.0
    freq_max: float = 20000.0
    bins_per_octave: int = 96
    time_resolution_ms: float = 10.0
    wavelet_bandwidth: float = 8.0
    wavelet_center_freq: float = 1.5


class UpdateSettingsPayload(PayloadBase):
    spectrogram: SpectrogramSettingsPayload = Field(default_factory=SpectrogramSettingsPayload)
    snap: SnapSettingsPayload = Field(default_factory=SnapSettingsPayload)


class TracePartialPayload(PayloadBase):
    trace: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time_direction(self) -> TracePartialPayload:
        direction = 0
        epsilon = 1e-9
        for index in range(1, len(self.trace)):
            delta = self.trace[index][0] - self.trace[index - 1][0]
            if abs(delta) <= epsilon:
                continue
            current = 1 if delta > 0 else -1
            if direction == 0:
                direction = current
                continue
            if current != direction:
                raise ValueError("Trace must move in one time direction.")
        return self


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
    mix_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    loop: bool = False
    start_position_sec: float | None = None
    speed_ratio: float = Field(default=1.0, ge=0.125, le=8.0)
    time_stretch_mode: str = Field(default="librosa", pattern="^(native|librosa)$")


class StopPayload(PayloadBase):
    return_position_sec: float | None = None


class HarmonicProbePayload(PayloadBase):
    time_sec: float


class MasterVolumePayload(PayloadBase):
    master_volume: float = Field(ge=0.0, le=1.0)


class UpdatePlaybackMixPayload(PayloadBase):
    mix_ratio: float = Field(ge=0.0, le=1.0)


class UpdatePlaybackSettingsPayload(PayloadBase):
    output_mode: str | None = Field(default=None, pattern="^(audio|midi)$")
    mix_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    speed_ratio: float | None = Field(default=None, ge=0.125, le=8.0)
    time_stretch_mode: str | None = Field(default=None, pattern="^(native|librosa)$")
    midi_mode: str | None = Field(default=None, pattern="^(mpe|multitrack|mono)$")
    midi_output_name: str | None = None
    midi_pitch_bend_range: int | None = Field(default=None, ge=1, le=96)
    midi_amplitude_mapping: str | None = Field(default=None, pattern="^(velocity|pressure|cc74|cc1)$")
    midi_amplitude_curve: str | None = Field(default=None, pattern="^(linear|db)$")
    midi_cc_update_rate_hz: int | None = Field(default=None, ge=1, le=1000)
    midi_bpm: float | None = Field(default=None, gt=0.0)


class ExportMpePayload(PayloadBase):
    pitch_bend_range: int = 48
    amplitude_mapping: str = "velocity"
    amplitude_curve: str = "linear"
    cc_update_rate_hz: int | None = Field(default=None, ge=1, le=1000)
    bpm: float = 120.0


class ExportMultiTrackMidiPayload(PayloadBase):
    pitch_bend_range: int = 12
    amplitude_mapping: str = "velocity"
    amplitude_curve: str = "linear"
    cc_update_rate_hz: int | None = Field(default=None, ge=1, le=1000)
    bpm: float = 120.0


class ExportMonophonicMidiPayload(PayloadBase):
    pitch_bend_range: int = 12
    amplitude_mapping: str = "velocity"
    amplitude_curve: str = "linear"
    cc_update_rate_hz: int | None = Field(default=None, ge=1, le=1000)
    bpm: float = 120.0


class ExportAudioPayload(PayloadBase):
    sample_rate: int | None = None
    bit_depth: int = 16
    output_type: str = "sine"
    cv_base_freq: float | None = None
    cv_full_scale_volts: float | None = None
    cv_mode: str = "mono"
    amplitude_curve: str = "linear"


class OpenAudioPathPayload(PayloadBase):
    path: str


class OpenAudioDataPayload(PayloadBase):
    name: str
    data_base64: str


class OpenProjectPathPayload(PayloadBase):
    path: str


class RequestSpectrogramTilePayload(PayloadBase):
    time_start: float
    time_end: float
    freq_min: float
    freq_max: float
    width: int
    height: int
    viewport_id: int | None = None
    gain: float | None = None
    min_db: float | None = None
    max_db: float | None = None
    gamma: float | None = None


class RequestSpectrogramOverviewPayload(PayloadBase):
    width: int = Field(default=2048, ge=16, le=4096)
    height: int = Field(default=320, ge=16, le=2048)
    gain: float | None = None
    min_db: float | None = None
    max_db: float | None = None
    gamma: float | None = None


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
