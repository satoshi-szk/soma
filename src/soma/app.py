from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, cast

import numpy as np
import webview

from soma.analysis import resample_audio
from soma.api_schema import (
    DeletePartialsPayload,
    ErasePartialPayload,
    ExportAudioPayload,
    ExportMpePayload,
    HitTestPayload,
    MergePartialsPayload,
    PayloadBase,
    PlayPayload,
    SelectInBoxPayload,
    ToggleMutePayload,
    TracePartialPayload,
    UpdatePartialPayload,
    UpdateSettingsPayload,
    parse_payload,
)
from soma.document import SomaDocument
from soma.exporter import AudioExportSettings, MpeExportSettings, export_audio, export_mpe
from soma.logging_utils import configure_logging, get_session_log_dir
from soma.models import AnalysisSettings, PartialPoint

logger = logging.getLogger(__name__)
_frontend_log_lock = threading.Lock()


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
    def __init__(self) -> None:
        self._doc = SomaDocument()
        self._last_audio_path: str | None = None
        self._frontend_log_path = get_session_log_dir("soma") / "frontend.log"

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def open_audio(self) -> dict[str, Any]:
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}

        result = window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=("Audio Files (*.wav;*.WAV)",),
        )
        if not result:
            return {"status": "cancelled"}

        selection = _normalize_dialog_result(result)
        if selection is None:
            return {"status": "cancelled"}
        path = Path(selection)
        try:
            info = self._doc.load_audio(path)
            self._doc.start_preview_async()
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("open_audio failed")
            return {"status": "error", "message": str(exc)}

        self._last_audio_path = info.path
        return {
            "status": "processing",
            "audio": info.to_dict(),
            "preview": None,
            "settings": self._doc.settings.to_dict(),
            "partials": [partial.to_dict() for partial in self._doc.store.all()],
        }

    def new_project(self) -> dict[str, Any]:
        self._doc.new_project()
        return {"status": "ok"}

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
        path = Path(selection)
        try:
            data = self._doc.load_project(path)
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
            info = self._doc.load_audio(audio_path, max_duration_sec=None)
        except Exception as exc:  # pragma: no cover
            logger.exception("open_project audio load failed")
            return {"status": "error", "message": str(exc)}

        self._doc.rebuild_resynth()
        self._doc.start_preview_async()
        return {
            "status": "processing",
            "audio": info.to_dict(),
            "preview": None,
            "settings": self._doc.settings.to_dict(),
            "partials": [partial.to_dict() for partial in self._doc.store.all()],
        }

    def save_project(self) -> dict[str, Any]:
        if self._doc.project_path is None:
            return self.save_project_as()
        try:
            self._doc.save_project(self._doc.project_path)
        except Exception as exc:  # pragma: no cover
            logger.exception("save_project failed")
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "path": str(self._doc.project_path)}

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
            self._doc.save_project(path)
        except Exception as exc:  # pragma: no cover
            logger.exception("save_project_as failed")
            return {"status": "error", "message": str(exc)}
        return {"status": "ok", "path": str(path)}

    def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = _validated_payload(UpdateSettingsPayload, payload, "update_settings")
        if isinstance(parsed, dict):
            return parsed
        settings = AnalysisSettings(**parsed.model_dump())
        self._doc.set_settings(settings)
        return {
            "status": "processing",
            "settings": self._doc.settings.to_dict(),
            "preview": None,
        }

    def trace_partial(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(TracePartialPayload, payload, "trace_partial")
            if isinstance(parsed, dict):
                return parsed
            points = [(point[0], point[1]) for point in parsed.trace]
            partial = self._doc.snap_partial(points)
            if partial is None:
                return {"status": "error", "message": "Failed to create partial."}
            return {"status": "ok", "partial": partial.to_dict()}
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
            self._doc.erase_path(points, radius_hz=radius)
            return {"status": "ok", "partials": [p.to_dict() for p in self._doc.store.all()]}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("erase_partial failed")
            return {"status": "error", "message": str(exc)}

    def update_partial(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(UpdatePartialPayload, payload, "update_partial")
            if isinstance(parsed, dict):
                return parsed
            points = [PartialPoint(time=p[0], freq=p[1], amp=p[2]) for p in parsed.points]
            partial = self._doc.update_partial_points(parsed.id, points)
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
            partial = self._doc.merge_partials(parsed.first, parsed.second)
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
            self._doc.delete_partials(parsed.ids)
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("delete_partials failed")
            return {"status": "error", "message": str(exc)}

    def toggle_mute(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(ToggleMutePayload, payload, "toggle_mute")
            if isinstance(parsed, dict):
                return parsed
            partial = self._doc.toggle_mute(parsed.id)
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
            result = self._doc.hit_test(parsed.time, parsed.freq, parsed.tolerance)
            return {"status": "ok", "id": result}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("hit_test failed")
            return {"status": "error", "message": str(exc)}

    def select_in_box(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(SelectInBoxPayload, payload, "select_in_box")
            if isinstance(parsed, dict):
                return parsed
            ids = self._doc.select_in_box(
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
            self._doc.undo()
            return {"status": "ok", "partials": [p.to_dict() for p in self._doc.store.all()]}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("undo failed")
            return {"status": "error", "message": str(exc)}

    def redo(self) -> dict[str, Any]:
        try:
            self._doc.redo()
            return {"status": "ok", "partials": [p.to_dict() for p in self._doc.store.all()]}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("redo failed")
            return {"status": "error", "message": str(exc)}

    def play(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _validated_payload(PlayPayload, payload, "play")
            if isinstance(parsed, dict):
                return parsed
            self._doc.play(parsed.mix_ratio, parsed.loop)
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("play failed")
            return {"status": "error", "message": str(exc)}

    def pause(self) -> dict[str, Any]:
        try:
            self._doc.pause()
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("pause failed")
            return {"status": "error", "message": str(exc)}

    def stop(self) -> dict[str, Any]:
        try:
            self._doc.stop()
            return {"status": "ok"}
        except Exception as exc:  # pragma: no cover - surface errors to UI
            logger.exception("stop failed")
            return {"status": "error", "message": str(exc)}

    def playback_state(self) -> dict[str, Any]:
        return {"status": "ok", "position": self._doc.playback_position()}

    def analysis_status(self) -> dict[str, Any]:
        state, preview, error = self._doc.get_preview_status()
        return {
            "status": "ok",
            "state": state,
            "preview": preview.to_dict() if preview else None,
            "message": error,
        }

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
            "is_playing": self._doc.is_playing(),
            "is_resynthesizing": self._doc.is_resynthesizing(),
            "position": self._doc.playback_position(),
        }

    def export_mpe(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._doc.audio_info is None:
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
        settings = MpeExportSettings(
            pitch_bend_range=parsed.pitch_bend_range,
            amplitude_mapping=parsed.amplitude_mapping,
            bpm=parsed.bpm,
        )
        paths = export_mpe(self._doc.store.all(), Path(selection), settings)
        return {"status": "ok", "paths": [str(p) for p in paths]}

    def export_audio(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._doc.audio_info is None:
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
        sample_rate = parsed.sample_rate if parsed.sample_rate is not None else self._doc.audio_info.sample_rate
        settings = AudioExportSettings(
            sample_rate=sample_rate,
            bit_depth=parsed.bit_depth,
            output_type=parsed.output_type,
        )
        buffer = self._doc.synth.get_mix_buffer().astype(np.float32)
        if settings.sample_rate != self._doc.audio_info.sample_rate:
            buffer, _ = resample_audio(buffer, self._doc.audio_info.sample_rate, settings.sample_rate)
        if settings.output_type == "cv":
            pitch, amp = self._doc.render_cv_buffers(settings.sample_rate)
            export_audio(
                Path(selection),
                buffer,
                settings,
                self._doc.settings.freq_min,
                self._doc.settings.freq_max,
                pitch_buffer=pitch,
                amp_buffer=amp,
            )
        else:
            export_audio(Path(selection), buffer, settings, self._doc.settings.freq_min, self._doc.settings.freq_max)
        return {"status": "ok", "path": str(selection)}


def _normalize_dialog_result(result: Any) -> str | None:
    if isinstance(result, (list, tuple)):
        return str(result[0]) if result else None
    if isinstance(result, str):
        return result
    return None


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
            return index_path.as_uri()

    repo_root = Path(__file__).resolve().parents[2]
    repo_index = repo_root / "frontend" / "dist" / "index.html"
    if repo_index.exists():
        return repo_index.as_uri()

    package_index = Path(__file__).resolve().parent / "ui" / "index.html"
    if package_index.exists():
        return package_index.as_uri()

    return "http://localhost:5173"


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
    configure_logging()
    force_dev = os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"}
    url = resolve_frontend_url()
    api = SomaApi()
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
    webview.start(debug=force_dev)


if __name__ == "__main__":
    main()
