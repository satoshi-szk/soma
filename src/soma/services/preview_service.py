from __future__ import annotations

import uuid
from collections.abc import Callable

import numpy as np

from soma.analysis import estimate_cwt_amp_reference
from soma.models import AnalysisSettings, Partial, PartialPoint, generate_bright_color
from soma.services.history import HistoryService
from soma.session import ProjectSession
from soma.workers import ComputeManager, SnapParams

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
        self._compute_manager = ComputeManager(snap_callback=self._on_snap_result)

    def cancel_all(self) -> None:
        self._compute_manager.cancel_all()

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
        with self._session._lock:
            self._session._snap_amp_reference = amp_reference
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
            min_freq = settings.snap.freq_min
            max_freq = settings.snap.freq_max
        freq_scale = 2.0**_SNAP_FREQ_MARGIN_OCTAVES
        freq_min = max(settings.snap.freq_min, min_freq / freq_scale)
        freq_max = min(settings.snap.freq_max, max_freq * freq_scale)
        freq_max = max(freq_max, freq_min * 1.001)

        with self._session._lock:
            fallback = self._session._snap_amp_reference

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
