from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from soma.models import (
    AnalysisSettings,
    Partial,
    PartialPoint,
    PlaybackSettings,
    ProjectMeta,
    SnapSettings,
    SourceInfo,
    SpectrogramSettings,
    generate_bright_color,
)

_MIDI_CC_UPDATE_RATE_OPTIONS_HZ = (50, 100, 200, 400, 800)


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_project_payload(
    source: SourceInfo,
    settings: AnalysisSettings,
    playback_settings: PlaybackSettings,
    partials: list[Partial],
    created_at: str | None = None,
) -> dict[str, Any]:
    meta = ProjectMeta(created_at=created_at or datetime.now(UTC).isoformat())
    return {
        "meta": meta.to_dict(),
        "source": source.to_dict(),
        "analysis_settings": settings.to_dict(),
        "playback_settings": playback_settings.to_dict(),
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
    if not isinstance(settings, dict):
        settings = {}

    spectrogram_raw = settings.get("spectrogram", settings)
    snap_raw = settings.get("snap", settings)
    if not isinstance(spectrogram_raw, dict):
        spectrogram_raw = {}
    if not isinstance(snap_raw, dict):
        snap_raw = {}

    legacy_freq_min = float(settings.get("freq_min", 20.0))
    legacy_freq_max = float(settings.get("freq_max", 20000.0))
    return AnalysisSettings(
        spectrogram=SpectrogramSettings(
            freq_min=float(spectrogram_raw.get("freq_min", legacy_freq_min)),
            freq_max=float(spectrogram_raw.get("freq_max", legacy_freq_max)),
            preview_freq_max=float(spectrogram_raw.get("preview_freq_max", 12000.0)),
            multires_blend_octaves=float(spectrogram_raw.get("multires_blend_octaves", 1.0)),
            gain=float(spectrogram_raw.get("gain", settings.get("gain", 1.0))),
            min_db=float(spectrogram_raw.get("min_db", settings.get("min_db", -80.0))),
            max_db=float(spectrogram_raw.get("max_db", settings.get("max_db", 0.0))),
            gamma=float(spectrogram_raw.get("gamma", settings.get("gamma", 1.0))),
        ),
        snap=SnapSettings(
            freq_min=float(snap_raw.get("freq_min", legacy_freq_min)),
            freq_max=float(snap_raw.get("freq_max", legacy_freq_max)),
            bins_per_octave=int(snap_raw.get("bins_per_octave", settings.get("bins_per_octave", 96))),
            time_resolution_ms=float(snap_raw.get("time_resolution_ms", settings.get("time_resolution_ms", 10.0))),
            wavelet_bandwidth=float(snap_raw.get("wavelet_bandwidth", settings.get("wavelet_bandwidth", 8.0))),
            wavelet_center_freq=float(snap_raw.get("wavelet_center_freq", settings.get("wavelet_center_freq", 1.5))),
        ),
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


def parse_playback_settings(data: dict[str, Any]) -> PlaybackSettings:
    settings = data.get("playback_settings", {})
    if not isinstance(settings, dict):
        return PlaybackSettings()
    master_volume = float(settings.get("master_volume", 1.0))
    output_mode = str(settings.get("output_mode", "audio"))
    mix_ratio = float(settings.get("mix_ratio", 0.55))
    speed_ratio = float(settings.get("speed_ratio", 1.0))
    time_stretch_mode = str(settings.get("time_stretch_mode", "librosa"))
    midi_mode = str(settings.get("midi_mode", "mpe"))
    midi_output_name = str(settings.get("midi_output_name", ""))
    midi_pitch_bend_range = int(settings.get("midi_pitch_bend_range", 48))
    midi_amplitude_mapping = str(settings.get("midi_amplitude_mapping", "cc74"))
    midi_amplitude_curve = str(settings.get("midi_amplitude_curve", "linear"))
    midi_cc_update_rate_hz = int(settings.get("midi_cc_update_rate_hz", 400))
    midi_cc_update_rate_hz = min(_MIDI_CC_UPDATE_RATE_OPTIONS_HZ, key=lambda rate: abs(rate - midi_cc_update_rate_hz))
    midi_bpm = float(settings.get("midi_bpm", 120.0))
    return PlaybackSettings(
        master_volume=min(1.0, max(0.0, master_volume)),
        output_mode="midi" if output_mode == "midi" else "audio",
        mix_ratio=min(1.0, max(0.0, mix_ratio)),
        speed_ratio=min(8.0, max(0.125, speed_ratio)),
        time_stretch_mode="native" if time_stretch_mode == "native" else "librosa",
        midi_mode=midi_mode if midi_mode in {"mpe", "multitrack", "mono"} else "mpe",
        midi_output_name=midi_output_name,
        midi_pitch_bend_range=max(1, midi_pitch_bend_range),
        midi_amplitude_mapping=(
            midi_amplitude_mapping if midi_amplitude_mapping in {"pressure", "cc74", "cc1", "velocity"} else "cc74"
        ),
        midi_amplitude_curve="db" if midi_amplitude_curve == "db" else "linear",
        midi_cc_update_rate_hz=midi_cc_update_rate_hz,
        midi_bpm=max(1.0, midi_bpm),
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
        color_raw = entry.get("color")
        color = _parse_color(color_raw)
        partials.append(
            Partial(
                id=str(entry.get("id", "")),
                points=points,
                is_muted=bool(entry.get("is_muted", False)),
                color=color,
            )
        )
    return partials


def _parse_color(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if re.fullmatch(r"#?[0-9a-fA-F]{6}", cleaned):
            return cleaned if cleaned.startswith("#") else f"#{cleaned}"
    return generate_bright_color()
