from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soma.analysis import estimate_cwt_amp_reference, make_spectrogram_stft
from soma.audio_utils import peak_normalize_buffer, time_stretch_pitch_preserving
from soma.exporter import MidiExportSettings, build_midi_for_playback
from soma.midi_player import MidiPlayer
from soma.models import (
    AnalysisSettings,
    AudioInfo,
    Partial,
    PartialPoint,
    PlaybackSettings,
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
    parse_playback_settings,
    parse_settings,
    parse_source,
)
from soma.synth import AudioPlayer, Synthesizer
from soma.workers import ComputeManager, SnapParams, ViewportParams

_CWT_PREVIEW_WINDOW_THRESHOLD_SEC = 2.0
_SNAP_TIME_MARGIN_SEC = 0.25
_SNAP_FREQ_MARGIN_OCTAVES = 0.5
_SNAP_QUEUE_MAX_WAITING = 2


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
        self.midi_player = MidiPlayer()
        self._lock = threading.Lock()
        self._is_resynthesizing = False
        self._logger = logging.getLogger(__name__)
        self._viewport_request_id: str | None = None
        self._viewport_preview: SpectrogramPreview | None = None
        self._event_sink = event_sink
        self._active_snap_request_id: str | None = None
        self._active_snap_trace: list[tuple[float, float]] | None = None
        self._queued_snaps: list[tuple[str, list[tuple[float, float]]]] = []
        self._playback_mode: str | None = None
        self._playback_output_mode = "audio"
        self._playback_speed_ratio = 1.0
        self._is_preparing_playback = False
        self._pending_playback_position_sec = 0.0
        self._playback_prepare_request_id = 0
        self._playback_content_revision = 0
        self._playback_cache_key: tuple[int, float, float, str] | None = None
        self._playback_cache_buffer: np.ndarray | None = None
        self._master_volume = 1.0
        self.player.set_master_volume(self._master_volume)
        self._last_mix_ratio = 0.55
        self._last_speed_ratio = 1.0
        self._last_time_stretch_mode = "librosa"
        self._midi_mode = "mpe"
        self._midi_output_name = ""
        self._midi_pitch_bend_range = 48
        self._midi_amplitude_mapping = "cc74"
        self._midi_amplitude_curve = "linear"
        self._midi_bpm = 120.0

        # バックグラウンド処理用の compute manager を初期化する。
        self._compute_manager = ComputeManager(
            viewport_callback=self._on_viewport_result,
            snap_callback=self._on_snap_result,
        )

    def new_project(self) -> None:
        # 保留中の計算をすべてキャンセルする。
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
        self._active_snap_request_id = None
        self._active_snap_trace = None
        self._queued_snaps = []
        self._playback_mode = None
        self._playback_output_mode = "audio"
        self._playback_speed_ratio = 1.0
        self._is_preparing_playback = False
        self._pending_playback_position_sec = 0.0
        self._playback_prepare_request_id += 1
        self._invalidate_playback_cache()
        self._last_mix_ratio = 0.55
        self._last_speed_ratio = 1.0
        self._last_time_stretch_mode = "librosa"
        self._midi_mode = "mpe"
        self._midi_output_name = ""
        self._midi_pitch_bend_range = 48
        self._midi_amplitude_mapping = "cc74"
        self._midi_amplitude_curve = "linear"
        self._midi_bpm = 120.0
        self.set_master_volume(1.0)

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

        # 振幅参照値を更新する。
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

        queued_snap: tuple[str, list[tuple[float, float]]] | None = None
        with self._lock:
            if request_id != self._active_snap_request_id:
                self._logger.debug(
                    "Ignoring stale snap result: %s (current: %s)",
                    request_id[:8] if request_id else "unknown",
                    self._active_snap_request_id[:8] if self._active_snap_request_id else "none",
                )
                return
            self._active_snap_request_id = None
            self._active_snap_trace = None
            if self._queued_snaps:
                next_request_id, next_trace = self._queued_snaps.pop(0)
                queued_snap = (next_request_id, list(next_trace))

        if error:
            self._logger.error("Snap computation error: %s", error)
            self._emit({"type": "snap_error", "request_id": request_id, "message": error})
        else:
            points_list = result.get("points", [])
            if len(points_list) < 2:
                self._logger.warning("Snap returned less than 2 points")
                self._emit({"type": "snap_error", "request_id": request_id, "message": "Failed to create partial"})
            else:
                # list 形式の点を PartialPoint オブジェクトへ変換する。
                snapped = [PartialPoint(time=p[0], freq=p[1], amp=p[2]) for p in points_list]

                # Partial を作成してストアへ追加する。
                partial_id = str(uuid.uuid4())
                partial = Partial(id=partial_id, points=snapped, color=generate_bright_color())
                before = self._snapshot_state(partial_ids=[partial_id])
                self.store.add(partial)
                self.synth.apply_partial(partial)
                after = self._snapshot_state(partial_ids=[partial_id])
                self.undo_manager.push(HistoryEntry(before=before, after=after))
                self._invalidate_playback_cache()

                self._logger.debug("Snap completed: partial_id=%s points=%d", partial_id[:8], len(snapped))
                self._emit({"type": "snap_completed", "request_id": request_id, "partial": partial.to_dict()})

        if queued_snap is not None:
            self._start_snap_request(queued_snap[0], queued_snap[1])

    def load_audio(
        self,
        path: Path,
        max_duration_sec: float | None = None,
        display_name: str | None = None,
    ) -> AudioInfo:
        from soma.analysis import load_audio

        info, audio = load_audio(path, max_duration_sec=max_duration_sec, display_name=display_name)
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
        self.midi_player.stop()
        self._playback_mode = None
        self._invalidate_playback_cache()
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

        start_now = False
        with self._lock:
            has_active = self._active_snap_request_id is not None
            if not has_active:
                start_now = True
            else:
                if len(self._queued_snaps) >= _SNAP_QUEUE_MAX_WAITING:
                    raise RuntimeError("Snap queue is full. Wait for current processing.")
                self._queued_snaps.append((request_id, list(trace)))
        if start_now:
            self._start_snap_request(request_id, trace)
        return request_id

    def _start_snap_request(self, request_id: str, trace: list[tuple[float, float]]) -> None:
        with self._lock:
            if self.audio_data is None or self.audio_info is None:
                if self._active_snap_request_id == request_id:
                    self._active_snap_request_id = None
                    self._active_snap_trace = None
                self._emit({"type": "snap_error", "request_id": request_id, "message": "No audio loaded."})
                return
            self._active_snap_request_id = request_id
            self._active_snap_trace = list(trace)
            audio = self.audio_data
            sample_rate = self.audio_info.sample_rate
            settings = self.settings

        params = self._build_snap_params(audio, sample_rate, settings, trace)
        self._logger.debug("Starting snap computation: %s", request_id[:8])
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

        with self._lock:
            fallback = self._amp_reference

        try:
            return estimate_cwt_amp_reference(
                audio=audio,
                sample_rate=sample_rate,
                settings=settings,
                freq_min=freq_min,
                freq_max=freq_max,
            )
        except Exception:  # pragma: no cover - keep snapping usable
            self._logger.exception("snap amp reference estimation failed")
            return fallback

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
        self._invalidate_playback_cache()
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
        self._invalidate_playback_cache()
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
        self._invalidate_playback_cache()

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
        self._invalidate_playback_cache()
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
        self._invalidate_playback_cache()
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
        self._playback_mode = None
        self._invalidate_playback_cache()

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

            # Overview は STFT のみを使う（CWT へのアップグレードなし）。
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

        # 入力パラメータを検証する。
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

        # CWT へのアップグレードを行うか判定する。
        window_duration = max(0.0, float(time_end - time_start))
        use_stft = window_duration > _CWT_PREVIEW_WINDOW_THRESHOLD_SEC

        width_clamped = int(np.clip(width, 16, 1536))
        height_clamped = int(np.clip(height, 16, 1024))

        params = ViewportParams(
            audio=self.audio_data,
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

    def play(
        self,
        mix_ratio: float,
        loop: bool,
        start_position_sec: float | None = None,
        speed_ratio: float = 1.0,
        time_stretch_mode: str = "librosa",
    ) -> None:
        if self.audio_info is None:
            return
        if self.is_resynthesizing():
            return
        self._last_mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        clamped_speed = float(np.clip(speed_ratio, 0.125, 8.0))
        self._last_speed_ratio = clamped_speed
        self._last_time_stretch_mode = time_stretch_mode
        self._cancel_playback_prepare()
        if self.player.is_playing():
            self.player.stop(reset_position_sec=None)
        if self.midi_player.is_playing():
            self.midi_player.stop()
        self._playback_mode = None
        self._playback_speed_ratio = 1.0
        start_sec = float(start_position_sec or 0.0)
        self._pending_playback_position_sec = start_sec
        self._playback_speed_ratio = clamped_speed
        if self._playback_output_mode == "midi":
            settings = MidiExportSettings(
                pitch_bend_range=self._midi_pitch_bend_range,
                amplitude_mapping=self._midi_amplitude_mapping,
                amplitude_curve=self._midi_amplitude_curve,
                bpm=self._midi_bpm,
            )
            midi = build_midi_for_playback(self.store.all(), self._midi_mode, settings)
            if not self._midi_output_name:
                raise ValueError("No MIDI output device selected.")
            self.midi_player.play_until(
                # MIDIイベントが途切れても、曲末（オーディオ尺）までは再生状態を維持する。
                midi=midi,
                output_name=self._midi_output_name,
                start_position_sec=start_sec,
                speed_ratio=clamped_speed,
                end_position_sec=self.audio_info.duration_sec,
            )
            self._playback_mode = "normal"
            return
        self._playback_mode = "normal"
        if abs(clamped_speed - 1.0) <= 1e-4:
            mixed = self._mix_buffer(self._last_mix_ratio)
            self.player.load(peak_normalize_buffer(mixed), self.audio_info.sample_rate)
            self.player.play(loop=loop, start_position_sec=start_sec)
            return
        cache_key = self._playback_cache_lookup_key(self._last_mix_ratio, clamped_speed, time_stretch_mode)
        cached = self._lookup_playback_cache(cache_key)
        if cached is not None:
            self.player.load(cached, self.audio_info.sample_rate)
            self.player.play(loop=loop, start_position_sec=start_sec / clamped_speed)
            return
        with self._lock:
            self._playback_prepare_request_id += 1
            request_id = self._playback_prepare_request_id
            self._is_preparing_playback = True
        sample_rate = self.audio_info.sample_rate

        def _worker() -> None:
            try:
                prepared = self._build_speed_adjusted_buffer(
                    self._last_mix_ratio, clamped_speed, sample_rate, time_stretch_mode
                )
                with self._lock:
                    if request_id != self._playback_prepare_request_id:
                        return
                    self._is_preparing_playback = False
                self._store_playback_cache(cache_key, prepared)
                self.player.load(prepared, sample_rate)
                self.player.play(loop=loop, start_position_sec=start_sec / clamped_speed)
            except Exception:
                self._logger.exception("playback prepare failed")
                with self._lock:
                    if request_id != self._playback_prepare_request_id:
                        return
                    self._is_preparing_playback = False
                    self._playback_mode = None
                    self._playback_speed_ratio = 1.0

        thread = threading.Thread(target=_worker, name="soma-playback-prepare", daemon=True)
        thread.start()

    def start_harmonic_probe(self, time_sec: float) -> bool:
        if self.audio_info is None:
            return False
        if self._playback_output_mode == "midi":
            return False
        self._cancel_playback_prepare()
        freqs, amps = self._harmonic_probe_tones(time_sec)
        if self.player.is_playing():
            self.player.stop(reset_position_sec=None)
        if self.midi_player.is_playing():
            self.midi_player.stop()
        started = self.player.play_probe(freqs, amps)
        if started:
            self._playback_mode = "probe"
            self._playback_speed_ratio = 1.0
        return started

    def update_harmonic_probe(self, time_sec: float) -> bool:
        if self._playback_mode != "probe":
            return False
        freqs, amps = self._harmonic_probe_tones(time_sec)
        return self.player.update_probe(freqs, amps)

    def stop_harmonic_probe(self) -> None:
        if self._playback_mode != "probe":
            return
        stopped = self.player.stop_probe()
        if not stopped:
            self.player.stop(reset_position_sec=None)
        self._playback_mode = None
        self._playback_speed_ratio = 1.0

    def pause(self) -> None:
        self._cancel_playback_prepare()
        self.player.pause()
        self.midi_player.stop()
        self._playback_mode = None
        self._playback_speed_ratio = 1.0

    def stop(self, return_position_sec: float | None = 0.0) -> None:
        self._cancel_playback_prepare()
        if self._playback_output_mode == "midi":
            self.midi_player.stop()
        else:
            reset_position = None if return_position_sec is None else return_position_sec / self._playback_speed_ratio
            self.player.stop(reset_position_sec=reset_position)
        self._playback_mode = None
        self._playback_speed_ratio = 1.0

    def set_master_volume(self, master_volume: float) -> float:
        clamped = float(np.clip(master_volume, 0.0, 1.0))
        self._master_volume = clamped
        self.player.set_master_volume(clamped)
        return clamped

    def master_volume(self) -> float:
        return self._master_volume

    def playback_settings(self) -> PlaybackSettings:
        return PlaybackSettings(
            master_volume=self._master_volume,
            output_mode=self._playback_output_mode,
            mix_ratio=self._last_mix_ratio,
            speed_ratio=self._last_speed_ratio,
            time_stretch_mode=self._last_time_stretch_mode,
            midi_mode=self._midi_mode,
            midi_output_name=self._midi_output_name,
            midi_pitch_bend_range=self._midi_pitch_bend_range,
            midi_amplitude_mapping=self._midi_amplitude_mapping,
            midi_amplitude_curve=self._midi_amplitude_curve,
            midi_bpm=self._midi_bpm,
        )

    def midi_outputs(self) -> list[str]:
        return self.midi_player.list_outputs()

    def update_playback_settings(self, settings: PlaybackSettings) -> PlaybackSettings:
        if settings.output_mode != self._playback_output_mode:
            self.stop(return_position_sec=None)
        self._playback_output_mode = settings.output_mode
        self._last_mix_ratio = float(np.clip(settings.mix_ratio, 0.0, 1.0))
        self._last_speed_ratio = float(np.clip(settings.speed_ratio, 0.125, 8.0))
        self._last_time_stretch_mode = settings.time_stretch_mode
        self._midi_mode = settings.midi_mode
        self._midi_output_name = settings.midi_output_name
        self._midi_pitch_bend_range = max(1, int(settings.midi_pitch_bend_range))
        self._midi_amplitude_mapping = settings.midi_amplitude_mapping
        self._midi_amplitude_curve = settings.midi_amplitude_curve
        self._midi_bpm = max(1.0, float(settings.midi_bpm))
        return self.playback_settings()

    def update_mix_ratio(self, mix_ratio: float) -> bool:
        if self.audio_info is None:
            return False
        clamped_mix = float(np.clip(mix_ratio, 0.0, 1.0))
        self._last_mix_ratio = clamped_mix
        if self._playback_output_mode == "midi":
            return True
        if self._playback_mode != "normal":
            return False
        if not self.player.is_playing():
            return False
        speed_ratio = self._playback_speed_ratio
        time_stretch_mode = self._last_time_stretch_mode
        sample_rate = self.audio_info.sample_rate
        if abs(speed_ratio - 1.0) <= 1e-4:
            updated = self._mix_buffer(clamped_mix)
            self.player.update_buffer(updated)
            return True
        updated = self._build_speed_adjusted_buffer(clamped_mix, speed_ratio, sample_rate, time_stretch_mode)
        self.player.update_buffer(updated)
        cache_key = self._playback_cache_lookup_key(clamped_mix, speed_ratio, time_stretch_mode)
        self._store_playback_cache(cache_key, updated)
        return True

    def playback_position(self) -> float:
        with self._lock:
            if self._is_preparing_playback:
                return self._pending_playback_position_sec
        if self._playback_output_mode == "midi":
            return self.midi_player.position_sec()
        return self.player.position_sec() * self._playback_speed_ratio

    def is_playing(self) -> bool:
        if self._playback_mode != "normal":
            return False
        if self._playback_output_mode == "midi":
            return self.midi_player.is_playing()
        return self.player.is_playing()

    def is_probe_playing(self) -> bool:
        return self._playback_mode == "probe" and self.player.is_playing()

    def is_preparing_playback(self) -> bool:
        with self._lock:
            return self._is_preparing_playback

    def _cancel_playback_prepare(self) -> None:
        with self._lock:
            self._playback_prepare_request_id += 1
            self._is_preparing_playback = False

    def _invalidate_playback_cache(self) -> None:
        with self._lock:
            self._playback_content_revision += 1
            self._playback_cache_key = None
            self._playback_cache_buffer = None

    def _playback_cache_lookup_key(
        self, mix_ratio: float, speed_ratio: float, mode: str
    ) -> tuple[int, float, float, str]:
        with self._lock:
            revision = self._playback_content_revision
        return (revision, round(float(mix_ratio), 4), round(float(speed_ratio), 6), mode)

    def _lookup_playback_cache(self, key: tuple[int, float, float, str]) -> np.ndarray | None:
        with self._lock:
            if self._playback_cache_key != key or self._playback_cache_buffer is None:
                return None
            return self._playback_cache_buffer

    def _store_playback_cache(self, key: tuple[int, float, float, str], buffer: np.ndarray) -> None:
        with self._lock:
            if key[0] != self._playback_content_revision:
                return
            self._playback_cache_key = key
            self._playback_cache_buffer = buffer.astype(np.float32, copy=False)

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
        source_info = self._prepare_source_info_for_save(path)
        payload = build_project_payload(
            source_info,
            self.settings,
            self.playback_settings(),
            self.store.all(),
        )
        from soma.persistence import save_project

        save_project(path, payload)
        self.project_path = path

    def _prepare_source_info_for_save(self, project_path: Path) -> SourceInfo:
        if self.source_info is None:
            raise ValueError("No source audio loaded")
        source_info = self.source_info
        source_path = Path(source_info.file_path).expanduser()
        if not source_path.is_absolute() and self.project_path is not None:
            source_path = (self.project_path.parent / source_path).resolve()

        requires_bundle = (
            not source_path.exists()
            or source_path.name.startswith("soma-drop-")
            or source_path.parent == Path(tempfile.gettempdir())
        )
        if not requires_bundle:
            return source_info
        if not source_path.exists():
            raise ValueError("Source audio file is missing and cannot be saved.")

        source_info = self._bundle_source_audio(project_path, source_path, source_info)
        self.source_info = source_info
        return source_info

    def _bundle_source_audio(self, project_path: Path, source_path: Path, source_info: SourceInfo) -> SourceInfo:
        bundle_dir = project_path.parent / f"{project_path.stem}_assets"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        preferred_name = self.audio_info.name if self.audio_info is not None else source_path.name
        sanitized = _sanitize_audio_filename(preferred_name, fallback=source_path.name)
        destination = _unique_destination(bundle_dir / sanitized)
        shutil.copy2(source_path, destination)
        relative_path = destination.relative_to(project_path.parent).as_posix()
        return SourceInfo(
            file_path=relative_path,
            sample_rate=source_info.sample_rate,
            duration_sec=source_info.duration_sec,
            md5_hash=source_info.md5_hash,
        )

    def load_project(self, path: Path) -> dict[str, Any]:
        data = load_project(path)
        source = parse_source(data)
        settings = parse_settings(data)
        playback_settings = parse_playback_settings(data)
        partials = parse_partials(data)
        self.settings = settings
        self.set_master_volume(playback_settings.master_volume)
        self.update_playback_settings(playback_settings)
        self.store = PartialStore()
        for partial in partials:
            self.store.add(partial)
        self.project_path = path
        self.source_info = source
        self.undo_manager.clear()
        return {"source": source, "settings": settings, "playback_settings": playback_settings, "partials": partials}

    def _mix_buffer(self, mix_ratio: float) -> np.ndarray:
        mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        resynth = self.synth.get_mix_buffer().astype(np.float32)
        if self.audio_data is None:
            return peak_normalize_buffer(resynth)
        original = self._match_length(self.audio_data, resynth.shape[0])
        mixed = (1.0 - mix_ratio) * original + mix_ratio * resynth
        return peak_normalize_buffer(mixed)

    def _match_length(self, audio: np.ndarray, length: int) -> np.ndarray:
        if audio.shape[0] == length:
            return audio.astype(np.float32)
        if audio.shape[0] < length:
            pad = np.zeros(length - audio.shape[0], dtype=np.float32)
            return np.concatenate([audio.astype(np.float32), pad])
        return audio[:length].astype(np.float32)

    def _build_speed_adjusted_buffer(
        self,
        mix_ratio: float,
        speed_ratio: float,
        sample_rate: int,
        time_stretch_mode: str,
    ) -> np.ndarray:
        if mix_ratio >= 1.0 - 1e-6:
            return self._render_resynth_time_scaled(speed_ratio, sample_rate)
        mixed = self._mix_buffer(mix_ratio)
        stretched = time_stretch_pitch_preserving(
            mixed,
            speed_ratio,
            sample_rate,
            mode=time_stretch_mode,
        )
        return peak_normalize_buffer(stretched)

    def _render_resynth_time_scaled(self, speed_ratio: float, sample_rate: int) -> np.ndarray:
        if self.audio_info is None:
            return np.zeros(1, dtype=np.float32)
        duration = self.audio_info.duration_sec
        total_samples = max(1, int((duration / speed_ratio) * sample_rate))
        buffer = np.zeros(total_samples, dtype=np.float64)
        fade_samples = max(1, int(sample_rate * 0.005))
        for partial in self.store.all():
            if partial.is_muted:
                continue
            points = partial.sorted_points()
            if len(points) < 2:
                continue
            times = np.array([point.time for point in points], dtype=np.float64)
            freqs = np.array([point.freq for point in points], dtype=np.float64)
            amps = np.array([point.amp for point in points], dtype=np.float64)
            start_scaled = max(0.0, float(times[0]) / speed_ratio)
            end_scaled = min(float(times[-1]) / speed_ratio, total_samples / sample_rate)
            start_idx = int(start_scaled * sample_rate)
            end_idx = int(end_scaled * sample_rate)
            if end_idx <= start_idx:
                continue
            scaled_times = (np.arange(end_idx - start_idx) / sample_rate) + start_scaled
            source_times = scaled_times * speed_ratio
            freq_interp = np.interp(source_times, times, freqs)
            amp_interp = np.interp(source_times, times, amps)
            phase = 2.0 * np.pi * np.cumsum(freq_interp) / sample_rate
            wave = np.sin(phase) * amp_interp
            fade = min(fade_samples, wave.size // 2)
            if fade > 0:
                fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float64)
                fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float64)
                wave[:fade] *= fade_in
                wave[-fade:] *= fade_out
            buffer[start_idx:end_idx] += wave
        return peak_normalize_buffer(buffer.astype(np.float32))

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
        self._invalidate_playback_cache()

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

    def _harmonic_probe_tones(self, time_sec: float) -> tuple[np.ndarray, np.ndarray]:
        freqs: list[float] = []
        amps: list[float] = []
        for partial in self.store.all():
            if partial.is_muted:
                continue
            sample = _partial_sample_at_time(partial, time_sec)
            if sample is None:
                continue
            freq, amp = sample
            if freq <= 0.0 or amp <= 0.0:
                continue
            freqs.append(freq)
            amps.append(amp)
        if not freqs:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        return np.asarray(freqs, dtype=np.float64), np.asarray(amps, dtype=np.float64)


def _ensure_soma_extension(path: Path) -> Path:
    if path.name.lower().endswith(".soma"):
        return path
    return path.with_name(f"{path.name}.soma")


def _sanitize_audio_filename(name: str, fallback: str) -> str:
    candidate = Path(name).name.strip()
    if not candidate or candidate in {".", ".."}:
        candidate = Path(fallback).name
    return candidate.replace("/", "_").replace("\\", "_")


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _partial_sample_at_time(partial: Partial, time_sec: float) -> tuple[float, float] | None:
    points = partial.sorted_points()
    if len(points) < 2:
        return None
    if time_sec < points[0].time or time_sec > points[-1].time:
        return None
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        if end.time < start.time:
            continue
        if time_sec < start.time or time_sec > end.time:
            continue
        dt = end.time - start.time
        if dt <= 1e-9:
            return end.freq, end.amp
        ratio = (time_sec - start.time) / dt
        freq = start.freq + (end.freq - start.freq) * ratio
        amp = start.amp + (end.amp - start.amp) * ratio
        return float(freq), float(amp)
    return float(points[-1].freq), float(points[-1].amp)


def _intersects_trace(partial: Partial, trace: list[tuple[float, float]], radius_hz: float) -> bool:
    return any(_point_in_erase_path(point, trace, radius_hz) for point in partial.points)


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
