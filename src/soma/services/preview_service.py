from __future__ import annotations

import uuid
from collections.abc import Callable

import numpy as np

from soma.analysis import estimate_cwt_amp_reference, make_spectrogram_stft
from soma.models import AnalysisSettings, Partial, PartialPoint, SpectrogramPreview, generate_bright_color
from soma.services.history import HistoryService
from soma.session import ProjectSession
from soma.workers import ComputeManager, SnapParams, ViewportParams

_CWT_PREVIEW_WINDOW_THRESHOLD_SEC = 2.0
_SNAP_TIME_MARGIN_SEC = 0.25
_SNAP_FREQ_MARGIN_OCTAVES = 0.5
_SNAP_QUEUE_MAX_WAITING = 2


class PreviewService:
    def __init__(
        self,
        session: ProjectSession,
        history: HistoryService,
        on_partials_changed: Callable[[], None],
    ) -> None:
        self._session = session
        self._history = history
        self._on_partials_changed = on_partials_changed
        self._compute_manager = ComputeManager(
            viewport_callback=self._on_viewport_result,
            snap_callback=self._on_snap_result,
        )

    def cancel_all(self) -> None:
        self._compute_manager.cancel_all()

    def start_preview_async(self) -> None:
        if self._session.audio_data is None or self._session.audio_info is None:
            return
        request_id = str(uuid.uuid4())
        with self._session._lock:
            self._session._preview_request_id = request_id
            self._session.preview_state = "processing"
            self._session.preview_error = None

            audio = self._session.audio_data
            sample_rate = self._session.audio_info.sample_rate
            settings = self._session.settings
            stft_reference = self._session._stft_amp_reference

        def _worker() -> None:
            with self._session._lock:
                if self._session._preview_request_id != request_id:
                    return
            duration = audio.shape[0] / float(sample_rate) if audio.size else 0.0
            freq_max = min(settings.preview_freq_max, settings.freq_max)

            try:
                preview, ref = make_spectrogram_stft(
                    audio=audio,
                    sample_rate=sample_rate,
                    settings=settings,
                    time_start=0.0,
                    time_end=duration,
                    freq_min=settings.freq_min,
                    freq_max=freq_max,
                    width=768,
                    height=320,
                    amp_reference=stft_reference,
                )
            except Exception as exc:  # pragma: no cover
                self._session._logger.exception("STFT preview generation failed")
                with self._session._lock:
                    if self._session._preview_request_id != request_id:
                        return
                    self._session.preview_state = "error"
                    self._session.preview_error = str(exc)
                self._session.emit_spectrogram_error("overview", str(exc))
                return

            with self._session._lock:
                if self._session._preview_request_id != request_id:
                    return
                self._session.preview = preview
                self._session._stft_amp_reference = ref
                self._session.preview_state = "ready"

            self._session.emit_spectrogram_preview("overview", "low", final=True, preview=preview)

        import threading

        thread = threading.Thread(target=_worker, name="soma-preview", daemon=True)
        thread.start()

    def get_preview_status(self) -> tuple[str, SpectrogramPreview | None, str | None]:
        with self._session._lock:
            return self._session.preview_state, self._session.preview, self._session.preview_error

    def start_viewport_preview_async(
        self,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
    ) -> str:
        if self._session.audio_data is None or self._session.audio_info is None:
            self._session._logger.debug("start_viewport_preview_async: no audio loaded")
            return ""

        duration = self._session.audio_data.shape[0] / float(self._session.audio_info.sample_rate)
        if time_start < 0 or time_end > duration + 0.01 or time_start >= time_end:
            self._session._logger.warning(
                "start_viewport_preview_async: invalid time range [%.3f, %.3f] for duration %.3f",
                time_start,
                time_end,
                duration,
            )
            return ""

        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            self._session._logger.warning(
                "start_viewport_preview_async: invalid dimensions %dx%d",
                width,
                height,
            )
            return ""

        request_id = str(uuid.uuid4())
        with self._session._lock:
            self._session._viewport_request_id = request_id
            self._session._viewport_preview = None

        window_duration = max(0.0, float(time_end - time_start))
        use_stft = window_duration > _CWT_PREVIEW_WINDOW_THRESHOLD_SEC

        width_clamped = int(np.clip(width, 16, 1536))
        height_clamped = int(np.clip(height, 16, 1024))

        params = ViewportParams(
            audio=self._session.audio_data,
            sample_rate=self._session.audio_info.sample_rate,
            settings=self._session.settings,
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
            width=width_clamped,
            height=height_clamped,
            use_stft=use_stft,
            stft_amp_reference=self._session._stft_amp_reference,
            cwt_amp_reference=self._session._amp_reference,
        )

        self._compute_manager.submit_viewport(request_id, params)
        return request_id

    def snap_partial_async(self, trace: list[tuple[float, float]]) -> str | None:
        if self._session.audio_data is None or self._session.audio_info is None:
            return None

        request_id = str(uuid.uuid4())

        start_now = False
        with self._session._lock:
            has_active = self._session._active_snap_request_id is not None
            if not has_active:
                start_now = True
            else:
                if len(self._session._queued_snaps) >= _SNAP_QUEUE_MAX_WAITING:
                    raise RuntimeError("Snap queue is full. Wait for current processing.")
                self._session._queued_snaps.append((request_id, list(trace)))
        if start_now:
            self._start_snap_request(request_id, trace)
        return request_id

    def _start_snap_request(self, request_id: str, trace: list[tuple[float, float]]) -> None:
        with self._session._lock:
            if self._session.audio_data is None or self._session.audio_info is None:
                if self._session._active_snap_request_id == request_id:
                    self._session._active_snap_request_id = None
                    self._session._active_snap_trace = None
                self._session.emit({"type": "snap_error", "request_id": request_id, "message": "No audio loaded."})
                return
            self._session._active_snap_request_id = request_id
            self._session._active_snap_trace = list(trace)
            audio = self._session.audio_data
            sample_rate = self._session.audio_info.sample_rate
            settings = self._session.settings

        params = self._build_snap_params(audio, sample_rate, settings, trace)
        self._compute_manager.submit_snap(request_id, params)

    def _build_snap_params(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: AnalysisSettings,
        trace: list[tuple[float, float]],
    ) -> SnapParams:
        start_sec, end_sec = self._snap_time_roi(trace, sample_rate, audio.shape[0])
        start_index = max(0, int(start_sec * sample_rate))
        end_index = min(audio.shape[0], int(end_sec * sample_rate))
        if end_index <= start_index:
            end_index = min(audio.shape[0], start_index + 1)
        local_audio = audio[start_index:end_index]
        time_offset_sec = start_index / float(sample_rate)
        local_trace = [(time_sec - time_offset_sec, freq_hz) for time_sec, freq_hz in trace]
        amp_reference = self._estimate_snap_amp_reference_for_trace(local_audio, sample_rate, settings, trace)
        return SnapParams(
            audio=local_audio,
            sample_rate=sample_rate,
            settings=settings,
            trace=local_trace,
            amp_reference=amp_reference,
            time_offset_sec=time_offset_sec,
        )

    def _snap_time_roi(
        self,
        trace: list[tuple[float, float]],
        sample_rate: int,
        total_samples: int,
    ) -> tuple[float, float]:
        if not trace:
            total_duration = total_samples / float(sample_rate)
            return 0.0, min(total_duration, _SNAP_TIME_MARGIN_SEC)
        total_duration = total_samples / float(sample_rate)
        min_time = min(point[0] for point in trace)
        max_time = max(point[0] for point in trace)
        start = max(0.0, min_time - _SNAP_TIME_MARGIN_SEC)
        end = min(total_duration, max_time + _SNAP_TIME_MARGIN_SEC)
        if end - start < 1.0 / float(sample_rate):
            end = min(total_duration, start + 1.0 / float(sample_rate))
        return start, end

    def _estimate_snap_amp_reference_for_trace(
        self,
        audio: np.ndarray,
        sample_rate: int,
        settings: AnalysisSettings,
        trace: list[tuple[float, float]],
    ) -> float | None:
        if audio.size == 0:
            return None
        if trace:
            min_freq = min(point[1] for point in trace)
            max_freq = max(point[1] for point in trace)
        else:
            min_freq = settings.freq_min
            max_freq = settings.freq_max
        freq_scale = 2.0**_SNAP_FREQ_MARGIN_OCTAVES
        freq_min = max(settings.freq_min, min_freq / freq_scale)
        freq_max = min(settings.freq_max, max_freq * freq_scale)
        freq_max = max(freq_max, freq_min * 1.001)

        with self._session._lock:
            fallback = self._session._amp_reference

        try:
            return estimate_cwt_amp_reference(
                audio=audio,
                sample_rate=sample_rate,
                settings=settings,
                freq_min=freq_min,
                freq_max=freq_max,
            )
        except Exception:  # pragma: no cover
            self._session._logger.exception("snap amp reference estimation failed")
            return fallback

    def _on_viewport_result(self, result: dict[str, object]) -> None:
        request_id = result.get("request_id")
        error = result.get("error")

        with self._session._lock:
            if request_id != self._session._viewport_request_id:
                return

        if error:
            self._session._logger.error("Viewport computation error: %s", error)
            self._session.emit_spectrogram_error("viewport", str(error))
            return

        preview_dict = result.get("preview")
        if preview_dict is None or not isinstance(preview_dict, dict):
            return

        quality = str(result.get("quality", "low"))
        final = bool(result.get("final", True))
        amp_reference = result.get("amp_reference")

        with self._session._lock:
            if (
                quality == "low"
                and isinstance(amp_reference, (int, float))
                and self._session._stft_amp_reference is None
            ):
                self._session._stft_amp_reference = float(amp_reference)
            elif quality == "high" and isinstance(amp_reference, (int, float)) and self._session._amp_reference is None:
                self._session._amp_reference = float(amp_reference)

        preview = SpectrogramPreview(
            width=int(preview_dict["width"]),
            height=int(preview_dict["height"]),
            data=list(preview_dict["data"]),
            freq_min=float(preview_dict["freq_min"]),
            freq_max=float(preview_dict["freq_max"]),
            duration_sec=float(preview_dict["duration_sec"]),
            time_start=float(preview_dict["time_start"]),
            time_end=float(preview_dict["time_end"]),
        )

        with self._session._lock:
            self._session._viewport_preview = preview

        self._session.emit_spectrogram_preview("viewport", quality, final, preview)

    def _on_snap_result(self, result: dict[str, object]) -> None:
        request_id = result.get("request_id")
        error = result.get("error")

        queued_snap: tuple[str, list[tuple[float, float]]] | None = None
        with self._session._lock:
            if request_id != self._session._active_snap_request_id:
                return
            self._session._active_snap_request_id = None
            self._session._active_snap_trace = None
            if self._session._queued_snaps:
                next_request_id, next_trace = self._session._queued_snaps.pop(0)
                queued_snap = (next_request_id, list(next_trace))

        if error:
            self._session._logger.error("Snap computation error: %s", error)
            self._session.emit({"type": "snap_error", "request_id": request_id, "message": str(error)})
        else:
            points_list = result.get("points", [])
            if not isinstance(points_list, list) or len(points_list) < 2:
                self._session._logger.warning("Snap returned less than 2 points")
                self._session.emit(
                    {"type": "snap_error", "request_id": request_id, "message": "Failed to create partial"}
                )
            else:
                snapped = [PartialPoint(time=p[0], freq=p[1], amp=p[2]) for p in points_list]
                partial_id = str(uuid.uuid4())
                partial = Partial(id=partial_id, points=snapped, color=generate_bright_color())
                before = self._history.snapshot_state(partial_ids=[partial_id])
                self._session.store.add(partial)
                self._session.synth.apply_partial(partial)
                after = self._history.snapshot_state(partial_ids=[partial_id])
                self._history.record(before, after)
                self._on_partials_changed()
                self._session.emit({"type": "snap_completed", "request_id": request_id, "partial": partial.to_dict()})

        if queued_snap is not None:
            self._start_snap_request(queued_snap[0], queued_snap[1])
