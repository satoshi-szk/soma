from __future__ import annotations

import base64
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, cast

import numpy as np
import webview

from soma.analysis import get_audio_duration_sec, resample_audio
from soma.api_schema import (
    DeletePartialsPayload,
    ErasePartialPayload,
    ExportAudioPayload,
    ExportMonophonicMidiPayload,
    ExportMpePayload,
    ExportMultiTrackMidiPayload,
    HarmonicProbePayload,
    HitTestPayload,
    MasterVolumePayload,
    MergePartialsPayload,
    OpenAudioDataPayload,
    OpenAudioPathPayload,
    OpenProjectPathPayload,
    PayloadBase,
    PlayPayload,
    RequestSpectrogramOverviewPayload,
    RequestSpectrogramTilePayload,
    SelectInBoxPayload,
    StopPayload,
    ToggleMutePayload,
    TracePartialPayload,
    UpdatePartialPayload,
    UpdatePlaybackMixPayload,
    UpdatePlaybackSettingsPayload,
    UpdateSettingsPayload,
    parse_payload,
)
from soma.cache_server import PreviewCacheServer
from soma.constants import MIDI_CC_UPDATE_RATE_OPTIONS_HZ
from soma.exporter import (
    AudioExportSettings,
    MonophonicExportSettings,
    MpeExportSettings,
    MultiTrackExportSettings,
    export_audio,
    export_cv_audio,
    export_monophonic_midi,
    export_mpe,
    export_multitrack_midi,
    render_cv_voice_buffers,
)
from soma.logging_utils import configure_logging, get_session_log_dir
from soma.models import AnalysisSettings, PartialPoint, PlaybackSettings, SnapSettings, SpectrogramSettings
from soma.preview_cache import PreviewCacheConfig, build_preview_payload
from soma.recent_projects import RecentProjectStore, default_recent_projects_path
from soma.services import HistoryService, PartialEditService, PlaybackService, PreviewService, ProjectService
from soma.session import ProjectSession

logger = logging.getLogger(__name__)
_frontend_log_lock = threading.Lock()
_frontend_event_lock = threading.Lock()
_preview_cache: PreviewCacheConfig | None = None
_preview_cache_server: PreviewCacheServer | None = None


def _dispatch_frontend_event(payload: dict[str, Any]) -> None:
    window = webview.windows[0] if webview.windows else None
    if window is None:
        return
    try:
        detail = json.dumps(payload, ensure_ascii=False)
        script = f"window.dispatchEvent(new CustomEvent('soma:event', {{ detail: {detail} }}));"
        with _frontend_event_lock:
            window.evaluate_js(script)
    except Exception:
        logger.exception("failed to dispatch frontend event")


def _summarize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        items = list(value.items())
        summary = {key: _summarize_payload(val) for key, val in items[:10]}
        if len(items) > 10:
            summary["..."] = f"{len(items) - 10} more"
        return summary
    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            preview = [_summarize_payload(item) for item in value[:3]]
            preview.append(f"...({len(value) - 3} more)")
            return preview
        return [_summarize_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes(len={len(value)})"
    if isinstance(value, str) and len(value) > 200:
        return f"{value[:200]}...(truncated)"
    return value


def _log_api_call(method: Any) -> Any:
    @wraps(method)
    def wrapper(self: SomaApi, *args: Any, **kwargs: Any) -> Any:
        logger.info(
            "api request %s args=%s kwargs=%s",
            method.__name__,
            _summarize_payload(args),
            _summarize_payload(kwargs),
        )
        result = method(self, *args, **kwargs)
        logger.info("api response %s result=%s", method.__name__, _summarize_payload(result))
        return result

    return wrapper


def _check_audio_duration(window: webview.Window, path: Path) -> dict[str, Any] | None:
    try:
        get_audio_duration_sec(path)
    except Exception:
        logger.exception("Failed to read audio header: %s", path)
        return {"status": "error", "message": "Could not read the audio file."}
    return None


def _validated_payload[PayloadT: PayloadBase](
    model: type[PayloadT],
    payload: dict[str, Any],
    method_name: str,
) -> PayloadT | dict[str, str]:
    parsed, error = parse_payload(model, payload)
    if error:
        logger.warning("api request %s invalid payload: %s", method_name, error)
        return {"status": "error", "message": error}
    if parsed is None:
        return {"status": "error", "message": "Invalid payload."}
    return cast(PayloadT, parsed)


class SomaApi:
    _MAX_ACTIVE_TILE_REQUESTS = 3

    def __init__(self) -> None:
        self._session = ProjectSession(event_sink=_dispatch_frontend_event)
        self._playback_service = PlaybackService(self._session)
        self._history_service = HistoryService(self._session)
        self._preview_service = PreviewService(
            self._session,
            self._history_service,
            on_partials_changed=self._playback_service.invalidate_cache,
        )
        self._history_service.set_callbacks(
            on_settings_applied=lambda: None,
            on_partials_changed=self._playback_service.invalidate_cache,
        )
        self._project_service = ProjectService(
            self._session,
            self._history_service,
            self._playback_service,
            self._preview_service,
        )
        self._partial_edit_service = PartialEditService(
            self._session,
            self._history_service,
            on_partials_changed=self._playback_service.invalidate_cache,
        )
        self._tile_request_lock = threading.Lock()
        self._active_tile_requests = 0
        self._latest_viewport_id = -1
        self._last_audio_path: str | None = None
        self._frontend_log_path = get_session_log_dir("soma") / "frontend.log"
        self._recent_projects = RecentProjectStore(default_recent_projects_path(), limit=10)

    def _playback_settings_payload(self) -> dict[str, Any]:
        return self._playback_service.playback_settings().to_dict()

    def _build_overview_preview(self, width: int = 768, height: int = 320) -> dict[str, Any] | None:
        if self._session.audio_data is None or self._session.audio_info is None:
            return None
        renderer = self._session._spectrogram_renderer
        if renderer is None:
            return None
        settings = self._session.settings
        try:
            preview, ref = renderer.render_overview(
                settings=settings.spectrogram,
                width=width,
                height=height,
                stft_amp_reference=self._session._stft_amp_reference,
            )
            self._session._stft_amp_reference = ref
        except Exception:
            logger.exception("build overview preview failed")
            return None
        return build_preview_payload(preview, _preview_cache, hint="overview")

    def _remember_recent_project(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            self._recent_projects.add(path)
        except Exception:
            logger.exception("failed to remember recent project: %s", path)

    def _project_load_response(self, info: Any) -> dict[str, Any]:
        preview_payload = self._build_overview_preview()
        if preview_payload is None:
            return {"status": "error", "message": "Failed to build overview preview."}
        return {
            "status": "ok",
            "audio": info.to_dict(),
            "preview": preview_payload,
            "settings": self._session.settings.to_dict(),
            "playback_settings": self._playback_settings_payload(),
            "partials": [partial.to_dict() for partial in self._session.store.all()],
        }

    def _open_project_from_path(self, path: Path) -> dict[str, Any]:
        try:
            data = self._project_service.load_project(path)
        except Exception as exc:  # pragma: no cover
            logger.exception("open_project load failed")
            return {"status": "error", "message": str(exc)}

        source = data.get("source")
        if source is None:
            return {"status": "error", "message": "Source audio not found in project."}
        audio_path = Path(source.file_path)
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()

        if not audio_path.exists():
            return {"status": "error", "message": f"Audio file missing: {audio_path}"}

        try:
            info = self._project_service.load_audio(audio_path, max_duration_sec=None)
        except Exception as exc:  # pragma: no cover
            logger.exception("open_project audio load failed")
            return {"status": "error", "message": str(exc)}

        self._playback_service.rebuild_resynth()
        self._remember_recent_project(path)
        return self._project_load_response(info)

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def open_audio(self) -> dict[str, Any]:
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}

        result = window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=("Audio Files (*.wav;*.WAV;*.aif;*.aiff;*.mp3;*.flac;*.ogg;*.m4a)",),
        )
        if not result:
            return {"status": "cancelled"}

        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        path = Path(selection)
        duration_check = _check_audio_duration(window, path)
        if duration_check is not None:
            return duration_check
        try:
            info = self._project_service.load_audio(path)
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("open_audio failed")
            return {"status": "error", "message": str(exc)}

        self._last_audio_path = info.path
        return {
            "status": "ok",
            "audio": info.to_dict(),
            "preview": self._build_overview_preview(),
            "settings": self._session.settings.to_dict(),
            "playback_settings": self._playback_settings_payload(),
            "partials": [partial.to_dict() for partial in self._session.store.all()],
        }

    def open_audio_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}

        parsed = _validated_payload(OpenAudioPathPayload, payload, "open_audio_path")
        if isinstance(parsed, dict):
            return parsed

        path = Path(parsed.path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists() or not path.is_file():
            return {"status": "error", "message": "Audio file not found."}

        duration_check = _check_audio_duration(window, path)
        if duration_check is not None:
            return duration_check
        try:
            info = self._project_service.load_audio(path)
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("open_audio_path failed")
            return {"status": "error", "message": str(exc)}

        self._last_audio_path = info.path
        return {
            "status": "ok",
            "audio": info.to_dict(),
            "preview": self._build_overview_preview(),
            "settings": self._session.settings.to_dict(),
            "playback_settings": self._playback_settings_payload(),
            "partials": [partial.to_dict() for partial in self._session.store.all()],
        }

    def open_audio_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}

        parsed = _validated_payload(OpenAudioDataPayload, payload, "open_audio_data")
        if isinstance(parsed, dict):
            return parsed

        suffix = Path(parsed.name).suffix or ".wav"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="soma-drop-") as handle:
                temp_path = Path(handle.name)
                handle.write(base64.b64decode(parsed.data_base64))
        except Exception as exc:  # pragma: no cover - unexpected decode errors
            logger.exception("open_audio_data failed to write temp file")
            return {"status": "error", "message": str(exc)}

        duration_check = _check_audio_duration(window, temp_path)
        if duration_check is not None:
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Failed to remove temp audio file: %s", temp_path)
            return duration_check
        try:
            info = self._project_service.load_audio(temp_path, display_name=parsed.name)
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("open_audio_data failed")
            return {"status": "error", "message": str(exc)}

        self._last_audio_path = info.path
        return {
            "status": "ok",
            "audio": info.to_dict(),
            "preview": self._build_overview_preview(),
            "settings": self._session.settings.to_dict(),
            "playback_settings": self._playback_settings_payload(),
            "partials": [partial.to_dict() for partial in self._session.store.all()],
        }

    def new_project(self) -> dict[str, Any]:
        self._project_service.new_project()
        return {"status": "ok", "playback_settings": self._playback_settings_payload()}

    def list_recent_projects(self) -> dict[str, Any]:
        rows = self._recent_projects.list()
        projects = [
            {
                "path": row["path"],
                "name": Path(row["path"]).name,
                "last_opened_at": row["last_opened_at"],
                "exists": Path(row["path"]).exists(),
            }
            for row in rows
        ]
        return {"status": "ok", "projects": projects}

    def open_project(self) -> dict[str, Any]:
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}

        result = window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=("SOMA Project (*.soma)",),
        )
        if not result:
            return {"status": "cancelled"}

        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        return self._open_project_from_path(Path(selection))

    def open_project_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = _validated_payload(OpenProjectPathPayload, payload, "open_project_path")
        if isinstance(parsed, dict):
            return parsed
        path = Path(parsed.path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.exists() or not path.is_file():
            return {"status": "error", "message": "Project file not found."}
        return self._open_project_from_path(path)

    def save_project(self) -> dict[str, Any]:
        if self._session.project_path is None:
            return self.save_project_as()
        try:
            self._project_service.save_project(self._session.project_path)
        except Exception as exc:  # pragma: no cover
            logger.exception("save_project failed")
            return {"status": "error", "message": str(exc)}
        self._remember_recent_project(self._session.project_path)
        return {"status": "ok", "path": str(self._session.project_path)}

    def save_project_as(self) -> dict[str, Any]:
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}
        result = window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename="project.soma",
            file_types=("SOMA Project (*.soma)",),
        )
        if not result:
            return {"status": "cancelled"}
        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        path = Path(selection)
        try:
            self._project_service.save_project(path)
        except Exception as exc:  # pragma: no cover
            logger.exception("save_project_as failed")
            return {"status": "error", "message": str(exc)}
        self._remember_recent_project(self._session.project_path)
        return {"status": "ok", "path": str(self._session.project_path)}

    def reveal_audio_in_explorer(self) -> dict[str, Any]:
        if self._session.audio_info is None:
            return {"status": "error", "message": "No audio loaded."}
        path = Path(self._session.audio_info.path).expanduser()
        if not path.exists():
            return {"status": "error", "message": "Audio file not found."}
        try:
            _reveal_in_file_explorer(path)
        except Exception as exc:  # pragma: no cover
            logger.exception("reveal_audio_in_explorer failed")
            return {"status": "error", "message": str(exc)}
        return {"status": "ok"}

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = _validated_payload(UpdateSettingsPayload, payload, "update_settings")
        if isinstance(parsed, dict):
            return parsed
        settings = AnalysisSettings(
            spectrogram=SpectrogramSettings(**parsed.spectrogram.model_dump()),
            snap=SnapSettings(**parsed.snap.model_dump()),
        )
        self._project_service.set_settings(settings)
        preview_payload = self._build_overview_preview()
        if preview_payload is None:
            return {"status": "error", "message": "Failed to build overview preview."}
        return {
            "status": "ok",
            "settings": self._session.settings.to_dict(),
            "preview": preview_payload,
        }

    def trace_partial(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Start async trace/snap computation. Result is sent via event."""
        try:
            parsed = _validated_payload(TracePartialPayload, payload, "trace_partial")
            if isinstance(parsed, dict):
                return parsed
            points = [(point[0], point[1]) for point in parsed.trace]
            request_id = self._preview_service.snap_partial_async(points)
            if request_id is None:
                logger.warning("trace_partial rejected: no audio loaded")
                return {"status": "error", "message": "No audio loaded."}
            return {"status": "accepted", "request_id": request_id}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("trace_partial failed")
            return {"status": "error", "message": str(exc)}

    def erase_partial(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(ErasePartialPayload, payload, "erase_partial")
            if isinstance(parsed, dict):
                return parsed
            points = [(point[0], point[1]) for point in parsed.trace]
            radius = parsed.radius_hz
            self._partial_edit_service.erase_path(points, radius_hz=radius)
            return {"status": "ok", "partials": [p.to_dict() for p in self._session.store.all()]}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("erase_partial failed")
            return {"status": "error", "message": str(exc)}

    def update_partial(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(UpdatePartialPayload, payload, "update_partial")
            if isinstance(parsed, dict):
                return parsed
            points = [PartialPoint(time=p[0], freq=p[1], amp=p[2]) for p in parsed.points]
            partial = self._partial_edit_service.update_partial_points(parsed.id, points)
            if partial is None:
                return {"status": "error", "message": "Partial not found."}
            return {"status": "ok", "partial": partial.to_dict()}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("update_partial failed")
            return {"status": "error", "message": str(exc)}

    def merge_partials(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(MergePartialsPayload, payload, "merge_partials")
            if isinstance(parsed, dict):
                return parsed
            partial = self._partial_edit_service.merge_partials(parsed.first, parsed.second)
            if partial is None:
                return {"status": "error", "message": "Failed to merge partials."}
            return {"status": "ok", "partial": partial.to_dict()}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("merge_partials failed")
            return {"status": "error", "message": str(exc)}

    def delete_partials(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(DeletePartialsPayload, payload, "delete_partials")
            if isinstance(parsed, dict):
                return parsed
            self._partial_edit_service.delete_partials(parsed.ids)
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("delete_partials failed")
            return {"status": "error", "message": str(exc)}

    def toggle_mute(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(ToggleMutePayload, payload, "toggle_mute")
            if isinstance(parsed, dict):
                return parsed
            partial = self._partial_edit_service.toggle_mute(parsed.id)
            if partial is None:
                return {"status": "error", "message": "Partial not found."}
            return {"status": "ok", "partial": partial.to_dict()}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("toggle_mute failed")
            return {"status": "error", "message": str(exc)}

    def hit_test(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(HitTestPayload, payload, "hit_test")
            if isinstance(parsed, dict):
                return parsed
            result = self._partial_edit_service.hit_test(parsed.time, parsed.freq, parsed.tolerance)
            return {"status": "ok", "id": result}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("hit_test failed")
            return {"status": "error", "message": str(exc)}

    def select_in_box(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(SelectInBoxPayload, payload, "select_in_box")
            if isinstance(parsed, dict):
                return parsed
            ids = self._partial_edit_service.select_in_box(
                time_start=parsed.time_start,
                time_end=parsed.time_end,
                freq_start=parsed.freq_start,
                freq_end=parsed.freq_end,
            )
            return {"status": "ok", "ids": ids}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("select_in_box failed")
            return {"status": "error", "message": str(exc)}

    def undo(self) -> dict[str, Any]:
        try:
            self._history_service.undo()
            return {"status": "ok", "partials": [p.to_dict() for p in self._session.store.all()]}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("undo failed")
            return {"status": "error", "message": str(exc)}

    def redo(self) -> dict[str, Any]:
        try:
            self._history_service.redo()
            return {"status": "ok", "partials": [p.to_dict() for p in self._session.store.all()]}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("redo failed")
            return {"status": "error", "message": str(exc)}

    def play(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(PlayPayload, payload, "play")
            if isinstance(parsed, dict):
                return parsed
            self._playback_service.play(
                parsed.mix_ratio,
                parsed.loop,
                parsed.start_position_sec,
                parsed.speed_ratio,
                parsed.time_stretch_mode,
            )
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("play failed")
            return {"status": "error", "message": str(exc)}

    def start_harmonic_probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(HarmonicProbePayload, payload, "start_harmonic_probe")
            if isinstance(parsed, dict):
                return parsed
            started = self._playback_service.start_harmonic_probe(parsed.time_sec)
            if not started:
                return {"status": "error", "message": "Failed to start harmonic probe."}
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("start_harmonic_probe failed")
            return {"status": "error", "message": str(exc)}

    def update_harmonic_probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(HarmonicProbePayload, payload, "update_harmonic_probe")
            if isinstance(parsed, dict):
                return parsed
            updated = self._playback_service.update_harmonic_probe(parsed.time_sec)
            if not updated:
                return {"status": "error", "message": "Failed to update harmonic probe."}
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("update_harmonic_probe failed")
            return {"status": "error", "message": str(exc)}

    def stop_harmonic_probe(self) -> dict[str, Any]:
        try:
            self._playback_service.stop_harmonic_probe()
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("stop_harmonic_probe failed")
            return {"status": "error", "message": str(exc)}

    def pause(self) -> dict[str, Any]:
        try:
            self._playback_service.pause()
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("pause failed")
            return {"status": "error", "message": str(exc)}

    def stop(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            stop_payload = payload or {}
            parsed = _validated_payload(StopPayload, stop_payload, "stop")
            if isinstance(parsed, dict):
                return parsed
            self._playback_service.stop(parsed.return_position_sec)
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("stop failed")
            return {"status": "error", "message": str(exc)}

    def set_master_volume(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(MasterVolumePayload, payload, "set_master_volume")
            if isinstance(parsed, dict):
                return parsed
            master_volume = self._playback_service.set_master_volume(parsed.master_volume)
            return {"status": "ok", "master_volume": master_volume}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("set_master_volume failed")
            return {"status": "error", "message": str(exc)}

    def update_playback_mix(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(UpdatePlaybackMixPayload, payload, "update_playback_mix")
            if isinstance(parsed, dict):
                return parsed
            updated = self._playback_service.update_mix_ratio(parsed.mix_ratio)
            if not updated:
                return {"status": "error", "message": "Playback mix update is not available now."}
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("update_playback_mix failed")
            return {"status": "error", "message": str(exc)}

    def list_midi_outputs(self) -> dict[str, Any]:
        outputs = self._playback_service.midi_outputs()
        error = self._playback_service.last_list_outputs_error()
        return {"status": "ok", "outputs": outputs, "error": error}

    def update_playback_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(UpdatePlaybackSettingsPayload, payload, "update_playback_settings")
            if isinstance(parsed, dict):
                return parsed
            current = self._playback_service.playback_settings()
            updated = self._playback_service.update_playback_settings(
                PlaybackSettings(
                    master_volume=current.master_volume,
                    output_mode=parsed.output_mode if parsed.output_mode is not None else current.output_mode,
                    mix_ratio=parsed.mix_ratio if parsed.mix_ratio is not None else current.mix_ratio,
                    speed_ratio=parsed.speed_ratio if parsed.speed_ratio is not None else current.speed_ratio,
                    time_stretch_mode=(
                        parsed.time_stretch_mode if parsed.time_stretch_mode is not None else current.time_stretch_mode
                    ),
                    midi_mode=parsed.midi_mode if parsed.midi_mode is not None else current.midi_mode,
                    midi_output_name=(
                        parsed.midi_output_name if parsed.midi_output_name is not None else current.midi_output_name
                    ),
                    midi_pitch_bend_range=(
                        parsed.midi_pitch_bend_range
                        if parsed.midi_pitch_bend_range is not None
                        else current.midi_pitch_bend_range
                    ),
                    midi_amplitude_mapping=(
                        parsed.midi_amplitude_mapping
                        if parsed.midi_amplitude_mapping is not None
                        else current.midi_amplitude_mapping
                    ),
                    midi_amplitude_curve=(
                        parsed.midi_amplitude_curve
                        if parsed.midi_amplitude_curve is not None
                        else current.midi_amplitude_curve
                    ),
                    midi_cc_update_rate_hz=(
                        parsed.midi_cc_update_rate_hz
                        if parsed.midi_cc_update_rate_hz is not None
                        else current.midi_cc_update_rate_hz
                    ),
                    midi_bpm=parsed.midi_bpm if parsed.midi_bpm is not None else current.midi_bpm,
                )
            )
            return {"status": "ok", "playback_settings": updated.to_dict()}
        except Exception as exc:  # pragma: no cover
            logger.exception("update_playback_settings failed")
            return {"status": "error", "message": str(exc)}

    def playback_state(self) -> dict[str, Any]:
        return {"status": "ok", "position": self._playback_service.playback_position()}

    def request_spectrogram_tile(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session.audio_data is None or self._session.audio_info is None:
            return {"status": "error", "message": "No audio loaded."}
        renderer = self._session._spectrogram_renderer
        if renderer is None:
            return {"status": "error", "message": "Renderer is not ready."}
        parsed = _validated_payload(RequestSpectrogramTilePayload, payload, "request_spectrogram_tile")
        if isinstance(parsed, dict):
            return parsed
        viewport_id = parsed.viewport_id
        with self._tile_request_lock:
            if viewport_id is not None and viewport_id > self._latest_viewport_id:
                self._latest_viewport_id = viewport_id
            if viewport_id is not None and viewport_id < self._latest_viewport_id:
                return {"status": "error", "message": "Stale tile request dropped."}
            if self._active_tile_requests >= self._MAX_ACTIVE_TILE_REQUESTS:
                return {"status": "error", "message": "Tile queue saturated. Retry latest viewport."}
            self._active_tile_requests += 1

        settings = self._session.settings
        try:
            if viewport_id is not None:
                with self._tile_request_lock:
                    if viewport_id < self._latest_viewport_id:
                        return {"status": "error", "message": "Stale tile request dropped."}
            if any(
                value is not None
                for value in (parsed.gain, parsed.min_db, parsed.max_db, parsed.gamma)
            ):
                spectrogram = settings.spectrogram
                settings = AnalysisSettings(
                    spectrogram=SpectrogramSettings(
                        **{
                            **spectrogram.to_dict(),
                            "gain": spectrogram.gain if parsed.gain is None else float(parsed.gain),
                            "min_db": spectrogram.min_db if parsed.min_db is None else float(parsed.min_db),
                            "max_db": spectrogram.max_db if parsed.max_db is None else float(parsed.max_db),
                            "gamma": spectrogram.gamma if parsed.gamma is None else float(parsed.gamma),
                        }
                    ),
                    snap=settings.snap,
                )

            preview, ref, quality = renderer.render_tile(
                settings=settings.spectrogram,
                time_start=parsed.time_start,
                time_end=parsed.time_end,
                freq_min=parsed.freq_min,
                freq_max=parsed.freq_max,
                width=parsed.width,
                height=parsed.height,
                stft_amp_reference=self._session._stft_amp_reference,
            )
            self._session._stft_amp_reference = ref
        except Exception as exc:
            logger.exception("request_spectrogram_tile failed")
            return {"status": "error", "message": str(exc)}
        finally:
            with self._tile_request_lock:
                self._active_tile_requests = max(0, self._active_tile_requests - 1)

        preview_payload = build_preview_payload(preview, _preview_cache, hint="tile")
        return {"status": "ok", "preview": preview_payload, "quality": quality}

    def request_spectrogram_overview(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session.audio_data is None or self._session.audio_info is None:
            return {"status": "error", "message": "No audio loaded."}
        renderer = self._session._spectrogram_renderer
        if renderer is None:
            return {"status": "error", "message": "Renderer is not ready."}
        parsed = _validated_payload(RequestSpectrogramOverviewPayload, payload, "request_spectrogram_overview")
        if isinstance(parsed, dict):
            return parsed
        settings = self._session.settings
        if any(
            value is not None
            for value in (parsed.gain, parsed.min_db, parsed.max_db, parsed.gamma)
        ):
            spectrogram = settings.spectrogram
            settings = AnalysisSettings(
                spectrogram=SpectrogramSettings(
                    **{
                        **spectrogram.to_dict(),
                        "gain": spectrogram.gain if parsed.gain is None else float(parsed.gain),
                        "min_db": spectrogram.min_db if parsed.min_db is None else float(parsed.min_db),
                        "max_db": spectrogram.max_db if parsed.max_db is None else float(parsed.max_db),
                        "gamma": spectrogram.gamma if parsed.gamma is None else float(parsed.gamma),
                    }
                ),
                snap=settings.snap,
            )
        try:
            preview, ref = renderer.render_overview(
                settings=settings.spectrogram,
                width=parsed.width,
                height=parsed.height,
                stft_amp_reference=self._session._stft_amp_reference,
            )
            self._session._stft_amp_reference = ref
        except Exception as exc:
            logger.exception("request_spectrogram_overview failed")
            return {"status": "error", "message": str(exc)}
        preview_payload = build_preview_payload(preview, _preview_cache, hint="overview")
        return {"status": "ok", "preview": preview_payload, "quality": "low"}

    def frontend_log(self, level: str, message: str) -> dict[str, Any]:
        timestamp = datetime.now(UTC).isoformat(timespec="milliseconds")
        line = f"{timestamp} {level.upper()} {message}".rstrip()
        with _frontend_log_lock:
            self._frontend_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._frontend_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return {"status": "ok"}

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "is_playing": self._playback_service.is_playing(),
            "is_probe_playing": self._playback_service.is_probe_playing(),
            "is_preparing_playback": self._playback_service.is_preparing_playback(),
            "is_resynthesizing": self._playback_service.is_resynthesizing(),
            "position": self._playback_service.playback_position(),
            "master_volume": self._playback_service.master_volume(),
            "playback_settings": self._playback_settings_payload(),
        }

    def export_mpe(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session.audio_info is None:
            return {"status": "error", "message": "No audio loaded."}
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}
        result = window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename="project.mid",
            file_types=("MIDI (*.mid)",),
        )
        if not result:
            return {"status": "cancelled"}
        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        parsed = _validated_payload(ExportMpePayload, payload, "export_mpe")
        if isinstance(parsed, dict):
            return parsed
        base_cc_rate = self._playback_service.playback_settings().midi_cc_update_rate_hz
        requested_cc_rate = parsed.cc_update_rate_hz if parsed.cc_update_rate_hz is not None else base_cc_rate
        cc_rate = min(MIDI_CC_UPDATE_RATE_OPTIONS_HZ, key=lambda rate: abs(rate - int(requested_cc_rate)))
        settings = MpeExportSettings(
            pitch_bend_range=parsed.pitch_bend_range,
            amplitude_mapping=parsed.amplitude_mapping,
            amplitude_curve=parsed.amplitude_curve,
            cc_update_rate_hz=cc_rate,
            bpm=parsed.bpm,
        )
        paths = export_mpe(self._session.store.all(), Path(selection), settings)
        return {"status": "ok", "paths": [str(p) for p in paths]}

    def export_multitrack_midi(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session.audio_info is None:
            return {"status": "error", "message": "No audio loaded."}
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}
        result = window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename="project_multitrack.mid",
            file_types=("MIDI (*.mid)",),
        )
        if not result:
            return {"status": "cancelled"}
        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        parsed = _validated_payload(ExportMultiTrackMidiPayload, payload, "export_multitrack_midi")
        if isinstance(parsed, dict):
            return parsed
        base_cc_rate = self._playback_service.playback_settings().midi_cc_update_rate_hz
        requested_cc_rate = parsed.cc_update_rate_hz if parsed.cc_update_rate_hz is not None else base_cc_rate
        cc_rate = min(MIDI_CC_UPDATE_RATE_OPTIONS_HZ, key=lambda rate: abs(rate - int(requested_cc_rate)))
        settings = MultiTrackExportSettings(
            pitch_bend_range=parsed.pitch_bend_range,
            amplitude_mapping=parsed.amplitude_mapping,
            amplitude_curve=parsed.amplitude_curve,
            cc_update_rate_hz=cc_rate,
            bpm=parsed.bpm,
        )
        paths = export_multitrack_midi(self._session.store.all(), Path(selection), settings)
        return {"status": "ok", "paths": [str(p) for p in paths]}

    def export_monophonic_midi(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session.audio_info is None:
            return {"status": "error", "message": "No audio loaded."}
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}
        result = window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename="project_mono.mid",
            file_types=("MIDI (*.mid)",),
        )
        if not result:
            return {"status": "cancelled"}
        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        parsed = _validated_payload(ExportMonophonicMidiPayload, payload, "export_monophonic_midi")
        if isinstance(parsed, dict):
            return parsed
        base_cc_rate = self._playback_service.playback_settings().midi_cc_update_rate_hz
        requested_cc_rate = parsed.cc_update_rate_hz if parsed.cc_update_rate_hz is not None else base_cc_rate
        cc_rate = min(MIDI_CC_UPDATE_RATE_OPTIONS_HZ, key=lambda rate: abs(rate - int(requested_cc_rate)))
        settings = MonophonicExportSettings(
            pitch_bend_range=parsed.pitch_bend_range,
            amplitude_mapping=parsed.amplitude_mapping,
            amplitude_curve=parsed.amplitude_curve,
            cc_update_rate_hz=cc_rate,
            bpm=parsed.bpm,
        )
        paths = export_monophonic_midi(self._session.store.all(), Path(selection), settings)
        return {"status": "ok", "paths": [str(p) for p in paths]}

    def export_audio(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session.audio_info is None:
            return {"status": "error", "message": "No audio loaded."}
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}
        result = window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename="project.wav",
            file_types=("WAV (*.wav)",),
        )
        if not result:
            return {"status": "cancelled"}
        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        parsed = _validated_payload(ExportAudioPayload, payload, "export_audio")
        if isinstance(parsed, dict):
            return parsed
        sample_rate = parsed.sample_rate if parsed.sample_rate is not None else self._session.audio_info.sample_rate
        settings = AudioExportSettings(
            sample_rate=sample_rate,
            bit_depth=parsed.bit_depth,
            output_type=parsed.output_type,
            cv_base_freq=parsed.cv_base_freq if parsed.cv_base_freq is not None else 440.0,
            cv_full_scale_volts=parsed.cv_full_scale_volts if parsed.cv_full_scale_volts is not None else 10.0,
            cv_mode=parsed.cv_mode,
            amplitude_curve=parsed.amplitude_curve,
        )
        buffer = self._playback_service.synth_mix_buffer()
        if settings.sample_rate != self._session.audio_info.sample_rate:
            buffer, _ = resample_audio(buffer, self._session.audio_info.sample_rate, settings.sample_rate)
        if settings.output_type == "cv":
            voice_buffers, amp_min, amp_max = render_cv_voice_buffers(
                self._session.store.all(),
                settings.sample_rate,
                self._session.audio_info.duration_sec,
                settings.cv_mode,
            )
            paths = export_cv_audio(Path(selection), settings, voice_buffers, amp_min, amp_max)
            return {"status": "ok", "path": str(paths[0]), "paths": [str(path) for path in paths]}
        else:
            export_audio(
                Path(selection),
                buffer,
                settings,
                self._session.settings.spectrogram.freq_min,
                self._session.settings.spectrogram.freq_max,
            )
        return {"status": "ok", "path": str(selection)}


def _normalize_dialog_result(result: Any) -> str | None:
    if isinstance(result, (list, tuple)):
        return str(result[0]) if result else None
    if isinstance(result, str):
        return result
    return None


def _reveal_in_file_explorer(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(resolved)], check=True)
        return
    if sys.platform.startswith("win"):
        subprocess.run(["explorer", "/select,", str(resolved)], check=True)
        return
    target = resolved.parent if resolved.is_file() else resolved
    subprocess.run(["xdg-open", str(target)], check=True)


def _wrap_api_methods() -> None:
    for name, attr in vars(SomaApi).items():
        if name.startswith("_"):
            continue
        if not callable(attr):
            continue
        if getattr(attr, "__wrapped__", None):
            continue
        setattr(SomaApi, name, _log_api_call(attr))


_wrap_api_methods()


def resolve_frontend_url() -> str:
    dev_url = os.environ.get("SOMA_DEV_SERVER_URL")
    if dev_url:
        return dev_url

    force_dev = os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"}
    if force_dev:
        return "http://localhost:5173"

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        index_path = Path(bundle_root) / "frontend" / "dist" / "index.html"
        if index_path.exists():
            return str(index_path)

    repo_root = Path(__file__).resolve().parents[2]
    repo_index = repo_root / "frontend" / "dist" / "index.html"
    if repo_index.exists():
        return str(repo_index)

    package_index = Path(__file__).resolve().parent / "ui" / "index.html"
    if package_index.exists():
        return str(package_index)

    return "http://localhost:5173"


def resolve_frontend_root() -> Path | None:
    dev_url = os.environ.get("SOMA_DEV_SERVER_URL")
    if dev_url:
        return None
    force_dev = os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"}
    if force_dev:
        return None
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        index_path = Path(bundle_root) / "frontend" / "dist" / "index.html"
        if index_path.exists():
            return index_path.parent
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_dist = repo_root / "frontend" / "dist"
    if repo_root_dist.exists():
        return repo_root_dist
    package_root = Path(__file__).resolve().parent / "ui"
    if package_root.exists():
        return package_root
    return None


def _configure_preview_cache() -> None:
    global _preview_cache
    global _preview_cache_server
    cache_dir = get_session_log_dir("soma") / "preview-cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_path = cache_dir / ".write-test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
    except Exception:
        logger.warning("preview cache disabled: cannot write to %s", cache_dir, exc_info=True)
        _preview_cache = None
        return
    if _preview_cache_server is None:
        _preview_cache_server = PreviewCacheServer(cache_dir)
    base_url = _preview_cache_server.start()
    _preview_cache = PreviewCacheConfig(dir_path=cache_dir, url_prefix=f"{base_url}/.soma-cache")


CONSOLE_HOOK_JS = r"""
(function () {
  function safeToString(v) {
    try {
      if (typeof v === 'string') return v;
      return JSON.stringify(v);
    } catch (e) {
      try { return String(v); } catch (_) { return '[unstringifiable]'; }
    }
  }

  function send(level, args) {
    try {
      if (window.pywebview && window.pywebview.api && window.pywebview.api.frontend_log) {
        const msg = args.map(safeToString).join(' ');
        window.pywebview.api.frontend_log(level, msg);
      }
    } catch (e) {
      // 送信失敗しても落とさない
    }
  }

  ['log', 'info', 'warn', 'error', 'debug'].forEach(function (level) {
    const orig = console[level];
    console[level] = function (...args) {
      send(level, args);
      return orig.apply(console, args);
    };
  });

  window.addEventListener('error', function (e) {
    const msg = (e && e.message ? e.message : 'Unknown error')
      + (e && e.filename ? ` @ ${e.filename}:${e.lineno}:${e.colno}` : '');
    const stack = (e && e.error && e.error.stack) ? ("\n" + e.error.stack) : "";
    send('error', [msg + stack]);
  });

  window.addEventListener('unhandledrejection', function (e) {
    const r = e && e.reason;
    const msg = r && r.stack ? r.stack : safeToString(r);
    send('error', ['UnhandledRejection: ' + msg]);
  });

  console.log('[ConsoleHook] Initialized');
})();
"""


def main() -> None:
    # macOS の PyInstaller 実行環境で multiprocessing を正しく動かすために必要。
    # これがないと子プロセスが main() を再実行し、無限にウィンドウが増殖する。
    multiprocessing.freeze_support()

    configure_logging()
    _configure_preview_cache()
    force_dev = os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"}
    url = resolve_frontend_url()
    api = SomaApi()
    if force_dev:
        # コンテキストメニューからは devtools を開けるようにしつつ、起動時の自動表示は抑止する。
        webview.settings["OPEN_DEVTOOLS_IN_DEBUG"] = False
    window = webview.create_window(
        "SOMA",
        url=url,
        js_api=api,
        width=1280,
        height=800,
        min_size=(960, 600),
        background_color="#f4f1ed",
    )
    if window is None:
        raise RuntimeError("Failed to create app window")

    def inject_console_hook() -> None:
        window.evaluate_js(CONSOLE_HOOK_JS)

    window.events.loaded += inject_console_hook
    webview.start(debug=force_dev, http_server=True)


if __name__ == "__main__":
    main()
