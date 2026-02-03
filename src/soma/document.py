from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soma.analysis import estimate_cwt_amp_reference, make_spectrogram_stft
from soma.models import (
    AnalysisSettings,
    AudioInfo,
    Partial,
    PartialPoint,
    SourceInfo,
    SpectrogramPreview,
    generate_bright_color,
)
from soma.partial_store import PartialStore
from soma.persistence import (
    build_project_payload,
    compute_md5,
    load_project,
    parse_partials,
    parse_settings,
    parse_source,
)
from soma.synth import AudioPlayer, Synthesizer
from soma.workers import ComputeManager, SnapParams, ViewportParams

_CWT_PREVIEW_WINDOW_THRESHOLD_SEC = 2.0


@dataclass
class ProjectState:
    audio_info: AudioInfo | None = None
    audio_data: np.ndarray | None = None
    settings: AnalysisSettings | None = None
    partials: dict[str, Partial | None] | None = None
    project_path: Path | None = None
    source_info: SourceInfo | None = None


@dataclass
class HistoryEntry:
    before: ProjectState
    after: ProjectState


class UndoRedoManager:
    def __init__(self) -> None:
        self._undo: list[HistoryEntry] = []
        self._redo: list[HistoryEntry] = []

    def push(self, entry: HistoryEntry) -> None:
        self._undo.append(entry)
        self._redo.clear()

    def undo(self) -> HistoryEntry | None:
        if not self._undo:
            return None
        return self._undo.pop()

    def redo(self) -> HistoryEntry | None:
        if not self._redo:
            return None
        return self._redo.pop()

    def push_redo(self, entry: HistoryEntry) -> None:
        self._redo.append(entry)

    def push_undo(self, entry: HistoryEntry) -> None:
        self._undo.append(entry)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


class SomaDocument:
    def __init__(self, event_sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.audio_info: AudioInfo | None = None
        self.audio_data: np.ndarray | None = None
        self.settings = AnalysisSettings()
        self.preview: SpectrogramPreview | None = None
        self._amp_reference: float | None = None
        self._stft_amp_reference: float | None = None
        self._snap_amp_reference: float | None = None
        self.preview_state = "idle"
        self.preview_error: str | None = None
        self.store = PartialStore()
        self.project_path: Path | None = None
        self.source_info: SourceInfo | None = None
        self.undo_manager = UndoRedoManager()
        self.synth = Synthesizer(sample_rate=44100, duration_sec=0.0)
        self.player = AudioPlayer()
        self._lock = threading.Lock()
        self._is_resynthesizing = False
        self._logger = logging.getLogger(__name__)
        self._viewport_request_id: str | None = None
        self._viewport_preview: SpectrogramPreview | None = None
        self._event_sink = event_sink
        self._pending_snap_request_id: str | None = None
        self._pending_snap_trace: list[tuple[float, float]] | None = None

        # Initialize compute manager for background processing
        self._compute_manager = ComputeManager(
            viewport_callback=self._on_viewport_result,
            snap_callback=self._on_snap_result,
        )

    def new_project(self) -> None:
        # Cancel any pending computations
        self._compute_manager.cancel_all()

        self.audio_info = None
        self.audio_data = None
        self.settings = AnalysisSettings()
        self.preview = None
        self._amp_reference = None
        self._stft_amp_reference = None
        self._snap_amp_reference = None
        self.store = PartialStore()
        self.project_path = None
        self.source_info = None
        self.undo_manager.clear()
        self.synth.reset(sample_rate=44100, duration_sec=0.0)
        self._viewport_request_id = None
        self._viewport_preview = None
        self._pending_snap_request_id = None
        self._pending_snap_trace = None

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(payload)
        except Exception:  # pragma: no cover
            self._logger.exception("event_sink failed")

    def _emit_spectrogram_preview(self, kind: str, quality: str, final: bool, preview: SpectrogramPreview) -> None:
        self._emit(
            {
                "type": "spectrogram_preview_updated",
                "kind": kind,
                "quality": quality,
                "final": final,
                "preview": preview.to_dict(),
            }
        )

    def _emit_spectrogram_error(self, kind: str, message: str) -> None:
        self._emit({"type": "spectrogram_preview_error", "kind": kind, "message": message})

    def _on_viewport_result(self, result: dict[str, Any]) -> None:
        """Callback for viewport worker results."""
        request_id = result.get("request_id")
        error = result.get("error")

        with self._lock:
            if request_id != self._viewport_request_id:
                self._logger.debug(
                    "Ignoring stale viewport result: %s (current: %s)",
                    request_id[:8] if request_id else "unknown",
                    self._viewport_request_id[:8] if self._viewport_request_id else "none",
                )
                return

        if error:
            self._logger.error("Viewport computation error: %s", error)
            self._emit_spectrogram_error("viewport", error)
            return

        preview_dict = result.get("preview")
        if preview_dict is None:
            return

        quality = result.get("quality", "low")
        final = result.get("final", True)
        amp_reference = result.get("amp_reference")

        # Update amp references
        with self._lock:
            if quality == "low" and amp_reference is not None and self._stft_amp_reference is None:
                self._stft_amp_reference = amp_reference
            elif quality == "high" and amp_reference is not None and self._amp_reference is None:
                self._amp_reference = amp_reference

        preview = SpectrogramPreview(
            width=preview_dict["width"],
            height=preview_dict["height"],
            data=preview_dict["data"],
            freq_min=preview_dict["freq_min"],
            freq_max=preview_dict["freq_max"],
            duration_sec=preview_dict["duration_sec"],
            time_start=preview_dict["time_start"],
            time_end=preview_dict["time_end"],
        )

        with self._lock:
            self._viewport_preview = preview

        self._logger.debug(
            "Viewport result received: request_id=%s quality=%s final=%s",
            request_id[:8] if request_id else "unknown",
            quality,
            final,
        )
        self._emit_spectrogram_preview("viewport", quality, final, preview)

    def _on_snap_result(self, result: dict[str, Any]) -> None:
        """Callback for snap worker results."""
        request_id = result.get("request_id")
        error = result.get("error")

        # Check if this result is for the current pending request
        with self._lock:
            if request_id != self._pending_snap_request_id:
                self._logger.debug(
                    "Ignoring stale snap result: %s (current: %s)",
                    request_id[:8] if request_id else "unknown",
                    self._pending_snap_request_id[:8] if self._pending_snap_request_id else "none",
                )
                return
            self._pending_snap_request_id = None
            self._pending_snap_trace = None

        if error:
            self._logger.error("Snap computation error: %s", error)
            self._emit({"type": "snap_error", "request_id": request_id, "message": error})
            return

        points_list = result.get("points", [])
        if len(points_list) < 2:
            self._logger.warning("Snap returned less than 2 points")
            self._emit({"type": "snap_error", "request_id": request_id, "message": "Failed to create partial"})
            return

        # Convert points from list format to PartialPoint objects
        snapped = [PartialPoint(time=p[0], freq=p[1], amp=p[2]) for p in points_list]

        # Create partial and add to store
        partial_id = str(uuid.uuid4())
        partial = Partial(id=partial_id, points=snapped, color=generate_bright_color())
        before = self._snapshot_state(partial_ids=[partial_id])
        self.store.add(partial)
        self.synth.apply_partial(partial)
        after = self._snapshot_state(partial_ids=[partial_id])
        self.undo_manager.push(HistoryEntry(before=before, after=after))

        self._logger.debug("Snap completed: partial_id=%s points=%d", partial_id[:8], len(snapped))
        self._emit({"type": "snap_completed", "request_id": request_id, "partial": partial.to_dict()})

    def load_audio(self, path: Path, max_duration_sec: float | None = None) -> AudioInfo:
        from soma.analysis import load_audio

        info, audio = load_audio(path, max_duration_sec)
        self.audio_info = info
        self.audio_data = audio
        self.source_info = SourceInfo(
            file_path=info.path,
            sample_rate=info.sample_rate,
            duration_sec=info.duration_sec,
            md5_hash=compute_md5(path),
        )
        self.preview = None
        self._amp_reference = None
        self._stft_amp_reference = None
        self._snap_amp_reference = None
        self.preview_state = "idle"
        self.preview_error = None
        self.synth.reset(sample_rate=info.sample_rate, duration_sec=info.duration_sec)
        self.player.load(self._mix_buffer(0.5), info.sample_rate)
        return info

    def set_settings(self, settings: AnalysisSettings) -> SpectrogramPreview | None:
        before = self._snapshot_state(include_settings=True)
        self.settings = settings
        self._snap_amp_reference = None
        if self.audio_data is None or self.audio_info is None:
            after = self._snapshot_state(include_settings=True)
            self.undo_manager.push(HistoryEntry(before=before, after=after))
            return None
        self.start_preview_async()
        after = self._snapshot_state(include_settings=True)
        self.undo_manager.push(HistoryEntry(before=before, after=after))
        return None

    def snap_partial_async(self, trace: list[tuple[float, float]]) -> str | None:
        """Start async snap computation. Returns request_id or None if no audio."""
        if self.audio_data is None or self.audio_info is None:
            return None

        request_id = str(uuid.uuid4())
        amp_reference = self._ensure_snap_amp_reference() or self._amp_reference

        with self._lock:
            self._pending_snap_request_id = request_id
            self._pending_snap_trace = trace

        params = SnapParams(
            audio=self.audio_data.copy(),  # Copy to avoid issues with shared memory
            sample_rate=self.audio_info.sample_rate,
            settings=self.settings,
            trace=trace,
            amp_reference=amp_reference,
        )

        self._logger.debug("Starting snap computation: %s", request_id[:8])
        self._compute_manager.submit_snap(request_id, params)
        return request_id

    def _ensure_snap_amp_reference(self) -> float | None:
        with self._lock:
            if self._snap_amp_reference is not None:
                return self._snap_amp_reference
            if self.audio_data is None or self.audio_info is None:
                return None
            audio = self.audio_data
            sample_rate = self.audio_info.sample_rate
            settings = self.settings

        try:
            reference = estimate_cwt_amp_reference(
                audio=audio,
                sample_rate=sample_rate,
                settings=settings,
                freq_min=settings.freq_min,
                freq_max=settings.freq_max,
            )
        except Exception:  # pragma: no cover - keep snapping usable
            self._logger.exception("snap amp reference estimation failed")
            return None

        with self._lock:
            if self.audio_data is audio and self._snap_amp_reference is None:
                self._snap_amp_reference = reference
            return self._snap_amp_reference

    def erase_path(self, trace: list[tuple[float, float]], radius_hz: float = 40.0) -> list[Partial]:
        if not trace:
            return []
        affected: list[Partial] = []
        for partial in self.store.all():
            if partial.is_muted:
                continue
            if _intersects_trace(partial, trace, radius_hz):
                affected.append(partial)
        if not affected:
            return []
        segments: list[Partial] = []
        for partial in affected:
            segments.extend(_split_partial(partial, trace, radius_hz))
        removed_ids = [p.id for p in affected]
        added_ids = [segment.id for segment in segments]
        tracked_ids = removed_ids + added_ids
        before = self._snapshot_state(partial_ids=tracked_ids)
        for partial in affected:
            self.store.remove(partial.id)
            self.synth.remove_partial(partial.id)
        for segment in segments:
            self.store.add(segment)
            self.synth.apply_partial(segment)
        after = self._snapshot_state(partial_ids=tracked_ids)
        self.undo_manager.push(HistoryEntry(before=before, after=after))
        return self.store.all()

    def toggle_mute(self, partial_id: str) -> Partial | None:
        partial = self.store.get(partial_id)
        if partial is None:
            return None
        before = self._snapshot_state(partial_ids=[partial_id])
        partial.is_muted = not partial.is_muted
        self.store.update(partial)
        self.synth.apply_partial(partial)
        after = self._snapshot_state(partial_ids=[partial_id])
        self.undo_manager.push(HistoryEntry(before=before, after=after))
        return partial

    def delete_partials(self, partial_ids: Iterable[str]) -> None:
        ids = list(partial_ids)
        if not ids:
            return
        before = self._snapshot_state(partial_ids=ids)
        for partial_id in ids:
            self.store.remove(partial_id)
            self.synth.remove_partial(partial_id)
        after = self._snapshot_state(partial_ids=ids)
        self.undo_manager.push(HistoryEntry(before=before, after=after))

    def update_partial_points(self, partial_id: str, points: list[PartialPoint]) -> Partial | None:
        partial = self.store.get(partial_id)
        if partial is None:
            return None
        before = self._snapshot_state(partial_ids=[partial_id])
        updated = Partial(id=partial_id, points=points, is_muted=partial.is_muted, color=partial.color)
        self.store.update(updated)
        self.synth.apply_partial(updated)
        after = self._snapshot_state(partial_ids=[partial_id])
        self.undo_manager.push(HistoryEntry(before=before, after=after))
        return updated

    def merge_partials(self, first_id: str, second_id: str) -> Partial | None:
        first = self.store.get(first_id)
        second = self.store.get(second_id)
        if first is None or second is None:
            return None
        merged_id = str(uuid.uuid4())
        tracked_ids = [first_id, second_id, merged_id]
        before = self._snapshot_state(partial_ids=tracked_ids)
        merged_points = sorted(first.points + second.points, key=lambda p: p.time)
        merged = Partial(id=merged_id, points=merged_points, color=generate_bright_color())
        self.store.remove(first_id)
        self.store.remove(second_id)
        self.synth.remove_partial(first_id)
        self.synth.remove_partial(second_id)
        self.store.add(merged)
        self.synth.apply_partial(merged)
        after = self._snapshot_state(partial_ids=tracked_ids)
        self.undo_manager.push(HistoryEntry(before=before, after=after))
        return merged

    def hit_test(self, time_sec: float, freq_hz: float, tolerance: float = 0.05) -> str | None:
        return self.store.hit_test(
            time_sec=time_sec,
            freq_hz=freq_hz,
            freq_min=self.settings.freq_min,
            freq_max=self.settings.freq_max,
            tolerance=tolerance,
        )

    def select_in_box(self, time_start: float, time_end: float, freq_start: float, freq_end: float) -> list[str]:
        return self.store.select_in_box(time_start, time_end, freq_start, freq_end)

    def undo(self) -> None:
        entry = self.undo_manager.undo()
        if entry is None:
            return
        self._apply_state(entry.before)
        self.undo_manager.push_redo(entry)

    def redo(self) -> None:
        entry = self.undo_manager.redo()
        if entry is None:
            return
        self._apply_state(entry.after)
        self.undo_manager.push_undo(entry)

    def is_resynthesizing(self) -> bool:
        with self._lock:
            return self._is_resynthesizing

    def rebuild_resynth(self) -> None:
        if self.audio_info is None:
            return
        with self._lock:
            self._is_resynthesizing = True
        self.synth.rebuild(self.store.all())
        with self._lock:
            self._is_resynthesizing = False
        self.player.load(self._mix_buffer(0.5), self.audio_info.sample_rate)

    def start_preview_async(self) -> None:
        """Generate overview preview using STFT only (fast, for base layer).

        This runs in a background thread but uses STFT which is fast enough
        that it won't significantly block. CWT is not used for overview.
        """
        if self.audio_data is None or self.audio_info is None:
            return
        request_id = str(uuid.uuid4())
        with self._lock:
            self._preview_request_id = request_id
            self.preview_state = "processing"
            self.preview_error = None

            audio = self.audio_data
            sample_rate = self.audio_info.sample_rate
            settings = self.settings
            stft_reference = self._stft_amp_reference

        def _worker() -> None:
            with self._lock:
                if self._preview_request_id != request_id:
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
            except Exception as exc:  # pragma: no cover - surface errors to UI
                self._logger.exception("STFT preview generation failed")
                with self._lock:
                    if self._preview_request_id != request_id:
                        return
                    self.preview_state = "error"
                    self.preview_error = str(exc)
                self._emit_spectrogram_error("overview", str(exc))
                return

            with self._lock:
                if self._preview_request_id != request_id:
                    return
                self.preview = preview
                self._stft_amp_reference = ref
                self.preview_state = "ready"

            # Overview uses STFT only (no CWT upgrade)
            self._emit_spectrogram_preview("overview", "low", final=True, preview=preview)

        thread = threading.Thread(target=_worker, name="soma-preview", daemon=True)
        thread.start()

    def get_preview_status(self) -> tuple[str, SpectrogramPreview | None, str | None]:
        with self._lock:
            return self.preview_state, self.preview, self.preview_error

    def start_viewport_preview_async(
        self,
        time_start: float,
        time_end: float,
        freq_min: float,
        freq_max: float,
        width: int,
        height: int,
    ) -> str:
        """Start async viewport preview computation using external process."""
        if self.audio_data is None or self.audio_info is None:
            self._logger.debug("start_viewport_preview_async: no audio loaded")
            return ""

        # Validate input parameters
        duration = self.audio_data.shape[0] / float(self.audio_info.sample_rate)
        if time_start < 0 or time_end > duration + 0.01 or time_start >= time_end:
            self._logger.warning(
                "start_viewport_preview_async: invalid time range [%.3f, %.3f] for duration %.3f",
                time_start,
                time_end,
                duration,
            )
            return ""

        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            self._logger.warning(
                "start_viewport_preview_async: invalid dimensions %dx%d",
                width,
                height,
            )
            return ""

        request_id = str(uuid.uuid4())
        self._logger.debug(
            "start_viewport_preview_async: request_id=%s time=[%.3f, %.3f] freq=[%.1f, %.1f] size=%dx%d",
            request_id[:8],
            time_start,
            time_end,
            freq_min,
            freq_max,
            width,
            height,
        )

        with self._lock:
            self._viewport_request_id = request_id
            self._viewport_preview = None

        # Determine if CWT upgrade should be performed
        window_duration = max(0.0, float(time_end - time_start))
        use_stft = window_duration > _CWT_PREVIEW_WINDOW_THRESHOLD_SEC

        width_clamped = int(np.clip(width, 16, 1536))
        height_clamped = int(np.clip(height, 16, 1024))

        params = ViewportParams(
            audio=self.audio_data.copy(),  # Copy to avoid shared memory issues
            sample_rate=self.audio_info.sample_rate,
            settings=self.settings,
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
            width=width_clamped,
            height=height_clamped,
            use_stft=use_stft,
            stft_amp_reference=self._stft_amp_reference,
            cwt_amp_reference=self._amp_reference,
        )

        self._compute_manager.submit_viewport(request_id, params)
        return request_id

    def play(self, mix_ratio: float, loop: bool, start_position_sec: float | None = None) -> None:
        if self.audio_info is None:
            return
        if self.is_resynthesizing():
            return
        self.player.load(self._mix_buffer(mix_ratio), self.audio_info.sample_rate)
        self.player.play(loop=loop, start_position_sec=start_position_sec or 0.0)

    def pause(self) -> None:
        self.player.pause()

    def stop(self, return_position_sec: float | None = 0.0) -> None:
        self.player.stop(reset_position_sec=return_position_sec)

    def playback_position(self) -> float:
        return self.player.position_sec()

    def is_playing(self) -> bool:
        return self.player.is_playing()

    def render_cv_buffers(self, sample_rate: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        if self.audio_info is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.0, 1.0
        duration = self.audio_info.duration_sec
        total_samples = max(1, int(duration * sample_rate))
        freq_buffer = np.zeros(total_samples, dtype=np.float32)
        amp_buffer = np.zeros(total_samples, dtype=np.float32)
        amp_min = float("inf")
        amp_max = float("-inf")
        for partial in self.store.all():
            if partial.is_muted:
                continue
            points = partial.sorted_points()
            if len(points) < 2:
                continue
            for point in points:
                if point.amp < amp_min:
                    amp_min = point.amp
                if point.amp > amp_max:
                    amp_max = point.amp
            times = np.array([p.time for p in points], dtype=np.float64)
            freqs = np.array([p.freq for p in points], dtype=np.float64)
            amps = np.array([p.amp for p in points], dtype=np.float64)
            start = max(0.0, float(times[0]))
            end = min(float(times[-1]), duration)
            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)
            if end_idx <= start_idx:
                continue
            sample_times = (np.arange(end_idx - start_idx) / sample_rate) + start
            freq_interp = np.interp(sample_times, times, freqs)
            amp_interp = np.interp(sample_times, times, amps)
            current = amp_buffer[start_idx:end_idx]
            mask = amp_interp > current
            current[mask] = amp_interp[mask].astype(np.float32)
            freq_buffer[start_idx:end_idx][mask] = freq_interp[mask].astype(np.float32)
        if amp_min == float("inf") or amp_max == float("-inf"):
            amp_min, amp_max = 0.0, 1.0
        elif amp_max <= amp_min:
            amp_max = amp_min + 1.0
        return freq_buffer, amp_buffer, float(amp_min), float(amp_max)

    def save_project(self, path: Path) -> None:
        if self.source_info is None or self.audio_info is None:
            raise ValueError("No audio loaded")
        path = _ensure_soma_extension(path)
        payload = build_project_payload(self.source_info, self.settings, self.store.all())
        from soma.persistence import save_project

        save_project(path, payload)
        self.project_path = path

    def load_project(self, path: Path) -> dict[str, Any]:
        data = load_project(path)
        source = parse_source(data)
        settings = parse_settings(data)
        partials = parse_partials(data)
        self.settings = settings
        self.store = PartialStore()
        for partial in partials:
            self.store.add(partial)
        self.project_path = path
        self.source_info = source
        self.undo_manager.clear()
        return {"source": source, "settings": settings, "partials": partials}

    def _mix_buffer(self, mix_ratio: float) -> np.ndarray:
        mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        resynth = self.synth.get_mix_buffer().astype(np.float32)
        if self.audio_data is None:
            return resynth
        original = self._match_length(self.audio_data, resynth.shape[0])
        return (1.0 - mix_ratio) * original + mix_ratio * resynth

    def _match_length(self, audio: np.ndarray, length: int) -> np.ndarray:
        if audio.shape[0] == length:
            return audio.astype(np.float32)
        if audio.shape[0] < length:
            pad = np.zeros(length - audio.shape[0], dtype=np.float32)
            return np.concatenate([audio.astype(np.float32), pad])
        return audio[:length].astype(np.float32)

    def _apply_state(self, state: ProjectState) -> None:
        if state.settings is not None:
            self.settings = state.settings
            if self.audio_data is not None and self.audio_info is not None:
                self.start_preview_async()
        if state.partials is None:
            return
        for partial_id, snapshot in state.partials.items():
            if snapshot is None:
                self.store.remove(partial_id)
                self.synth.remove_partial(partial_id)
                continue
            self.store.update(snapshot)
            self.synth.apply_partial(snapshot)

    def _snapshot_state(
        self,
        *,
        partial_ids: Iterable[str] | None = None,
        include_settings: bool = False,
    ) -> ProjectState:
        snapshot: dict[str, Partial | None] | None = None
        if partial_ids is not None:
            snapshot = {}
            for partial_id in partial_ids:
                partial = self.store.get(partial_id)
                if partial is None:
                    snapshot[partial_id] = None
                else:
                    snapshot[partial_id] = Partial(
                        id=partial.id,
                        points=list(partial.points),
                        is_muted=partial.is_muted,
                        color=partial.color,
                    )
        return ProjectState(
            audio_info=self.audio_info,
            audio_data=self.audio_data,
            settings=self.settings if include_settings else None,
            partials=snapshot,
            project_path=self.project_path,
            source_info=self.source_info,
        )


def _ensure_soma_extension(path: Path) -> Path:
    if path.name.lower().endswith(".soma"):
        return path
    return path.with_name(f"{path.name}.soma")


def _intersects_trace(partial: Partial, trace: list[tuple[float, float]], radius_hz: float) -> bool:
    for point in partial.points:
        if _point_in_erase_path(point, trace, radius_hz):
            return True
    return False


def _point_in_erase_path(point: PartialPoint, trace: list[tuple[float, float]], radius_hz: float) -> bool:
    for time_sec, freq_hz in trace:
        if abs(point.time - time_sec) <= 0.02 and abs(point.freq - freq_hz) <= radius_hz:
            return True
    return False


def _split_partial(
    partial: Partial,
    trace: list[tuple[float, float]],
    radius_hz: float,
) -> list[Partial]:
    remaining = [p for p in partial.points if not _point_in_erase_path(p, trace, radius_hz)]
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
