from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soma.analysis import make_spectrogram_preview, snap_trace
from soma.models import AnalysisSettings, AudioInfo, Partial, PartialPoint, SourceInfo, SpectrogramPreview
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


@dataclass
class ProjectState:
    audio_info: AudioInfo | None = None
    audio_data: np.ndarray | None = None
    settings: AnalysisSettings | None = None
    partials: dict[str, Partial | None] | None = None
    project_path: Path | None = None
    source_info: SourceInfo | None = None


class UndoRedoManager:
    def __init__(self) -> None:
        self._undo: list[ProjectState] = []
        self._redo: list[ProjectState] = []

    def push(self, state: ProjectState) -> None:
        self._undo.append(state)
        self._redo.clear()

    def undo(self) -> ProjectState | None:
        if not self._undo:
            return None
        state = self._undo.pop()
        return state

    def redo(self) -> ProjectState | None:
        if not self._redo:
            return None
        state = self._redo.pop()
        return state

    def push_redo(self, state: ProjectState) -> None:
        self._redo.append(state)

    def push_undo(self, state: ProjectState) -> None:
        self._undo.append(state)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


class SomaDocument:
    def __init__(self) -> None:
        self.audio_info: AudioInfo | None = None
        self.audio_data: np.ndarray | None = None
        self.settings = AnalysisSettings()
        self.preview: SpectrogramPreview | None = None
        self.preview_state = "idle"
        self.preview_error: str | None = None
        self._preview_thread: threading.Thread | None = None
        self.store = PartialStore()
        self.project_path: Path | None = None
        self.source_info: SourceInfo | None = None
        self.undo_manager = UndoRedoManager()
        self.synth = Synthesizer(sample_rate=44100, duration_sec=0.0)
        self.player = AudioPlayer()
        self._lock = threading.Lock()
        self._is_resynthesizing = False
        self._logger = logging.getLogger(__name__)

    def new_project(self) -> None:
        self.audio_info = None
        self.audio_data = None
        self.settings = AnalysisSettings()
        self.preview = None
        self.store = PartialStore()
        self.project_path = None
        self.source_info = None
        self.undo_manager.clear()
        self.synth.reset(sample_rate=44100, duration_sec=0.0)

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
        self.preview_state = "idle"
        self.preview_error = None
        self.synth.reset(sample_rate=info.sample_rate, duration_sec=info.duration_sec)
        self.player.load(self._mix_buffer(0.5), info.sample_rate)
        return info

    def set_settings(self, settings: AnalysisSettings) -> SpectrogramPreview | None:
        self._push_settings()
        self.settings = settings
        if self.audio_data is None or self.audio_info is None:
            return None
        self.start_preview_async()
        return None

    def snap_partial(self, trace: list[tuple[float, float]]) -> Partial | None:
        if self.audio_data is None or self.audio_info is None:
            return None
        snapped = snap_trace(self.audio_data, self.audio_info.sample_rate, self.settings, trace)
        if len(snapped) < 2:
            return None
        partial_id = str(uuid.uuid4())
        partial = Partial(id=partial_id, points=snapped)
        self._push_state([partial_id])
        self.store.add(partial)
        self.synth.apply_partial(partial)
        return partial

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
        removed_ids = [p.id for p in affected]
        self._push_state(removed_ids)
        for partial in affected:
            self.store.remove(partial.id)
            self.synth.remove_partial(partial.id)
            for segment in _split_partial(partial, trace, radius_hz):
                self.store.add(segment)
                self.synth.apply_partial(segment)
        return self.store.all()

    def toggle_mute(self, partial_id: str) -> Partial | None:
        partial = self.store.get(partial_id)
        if partial is None:
            return None
        self._push_state([partial_id])
        partial.is_muted = not partial.is_muted
        self.store.update(partial)
        self.synth.apply_partial(partial)
        return partial

    def delete_partials(self, partial_ids: Iterable[str]) -> None:
        ids = list(partial_ids)
        if not ids:
            return
        self._push_state(ids)
        for partial_id in ids:
            self.store.remove(partial_id)
            self.synth.remove_partial(partial_id)

    def update_partial_points(self, partial_id: str, points: list[PartialPoint]) -> Partial | None:
        partial = self.store.get(partial_id)
        if partial is None:
            return None
        self._push_state([partial_id])
        updated = Partial(id=partial_id, points=points, is_muted=partial.is_muted)
        self.store.update(updated)
        self.synth.apply_partial(updated)
        return updated

    def merge_partials(self, first_id: str, second_id: str) -> Partial | None:
        first = self.store.get(first_id)
        second = self.store.get(second_id)
        if first is None or second is None:
            return None
        self._push_state([first_id, second_id])
        merged_points = sorted(first.points + second.points, key=lambda p: p.time)
        merged = Partial(id=str(uuid.uuid4()), points=merged_points)
        self.store.remove(first_id)
        self.store.remove(second_id)
        self.synth.remove_partial(first_id)
        self.synth.remove_partial(second_id)
        self.store.add(merged)
        self.synth.apply_partial(merged)
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
        state = self.undo_manager.undo()
        if state is None:
            return
        redo_state = self._snapshot_state(state)
        self._apply_state(state)
        self.undo_manager.push_redo(redo_state)

    def redo(self) -> None:
        state = self.undo_manager.redo()
        if state is None:
            return
        undo_state = self._snapshot_state(state)
        self._apply_state(state)
        self.undo_manager.push_undo(undo_state)

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
        if self.audio_data is None or self.audio_info is None:
            return
        with self._lock:
            if self._preview_thread and self._preview_thread.is_alive():
                return
            self.preview_state = "processing"
            self.preview_error = None

            audio = self.audio_data
            sample_rate = self.audio_info.sample_rate
            settings = self.settings

        def _worker() -> None:
            try:
                preview = make_spectrogram_preview(audio, sample_rate, settings)
            except Exception as exc:  # pragma: no cover - surface in UI via status
                self._logger.exception("preview generation failed")
                with self._lock:
                    self.preview_state = "error"
                    self.preview_error = str(exc)
                return
            with self._lock:
                self.preview = preview
                self.preview_state = "ready"

        thread = threading.Thread(target=_worker, name="soma-preview", daemon=True)
        with self._lock:
            self._preview_thread = thread
        thread.start()

    def get_preview_status(self) -> tuple[str, SpectrogramPreview | None, str | None]:
        with self._lock:
            return self.preview_state, self.preview, self.preview_error

    def play(self, mix_ratio: float, loop: bool) -> None:
        if self.audio_info is None:
            return
        if self.is_resynthesizing():
            return
        self.player.load(self._mix_buffer(mix_ratio), self.audio_info.sample_rate)
        self.player.play(loop=loop)

    def pause(self) -> None:
        self.player.pause()

    def stop(self) -> None:
        self.player.stop()

    def playback_position(self) -> float:
        return self.player.position_sec()

    def is_playing(self) -> bool:
        return self.player.is_playing()

    def render_cv_buffers(self, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        if self.audio_info is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        duration = self.audio_info.duration_sec
        total_samples = max(1, int(duration * sample_rate))
        freq_buffer = np.zeros(total_samples, dtype=np.float32)
        amp_buffer = np.zeros(total_samples, dtype=np.float32)
        for partial in self.store.all():
            if partial.is_muted:
                continue
            points = partial.sorted_points()
            if len(points) < 2:
                continue
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
        return freq_buffer, amp_buffer

    def save_project(self, path: Path) -> None:
        if self.source_info is None or self.audio_info is None:
            raise ValueError("No audio loaded")
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

    def _push_state(self, partial_ids: Iterable[str]) -> None:
        state = self._snapshot_state(ProjectState(partials={pid: None for pid in partial_ids}))
        self.undo_manager.push(state)

    def _apply_state(self, state: ProjectState) -> None:
        if state.settings is not None:
            self.settings = state.settings
            if self.audio_data is not None and self.audio_info is not None:
                self.preview = make_spectrogram_preview(
                    self.audio_data, self.audio_info.sample_rate, self.settings
                )
        if state.partials is None:
            return
        for partial_id, snapshot in state.partials.items():
            if snapshot is None:
                self.store.remove(partial_id)
                self.synth.remove_partial(partial_id)
                continue
            self.store.update(snapshot)
            self.synth.apply_partial(snapshot)

    def _snapshot_state(self, reference: ProjectState) -> ProjectState:
        snapshot: dict[str, Partial | None] = {}
        if reference.partials is None:
            return reference
        for partial_id in reference.partials:
            partial = self.store.get(partial_id)
            if partial is None:
                snapshot[partial_id] = None
            else:
                snapshot[partial_id] = Partial(
                    id=partial.id,
                    points=list(partial.points),
                    is_muted=partial.is_muted,
                )
        return ProjectState(
            audio_info=self.audio_info,
            audio_data=self.audio_data,
            settings=self.settings,
            partials=snapshot,
            project_path=self.project_path,
            source_info=self.source_info,
        )

    def _push_settings(self) -> None:
        state = ProjectState(settings=self.settings, partials={})
        self.undo_manager.push(state)


def _intersects_trace(partial: Partial, trace: list[tuple[float, float]], radius_hz: float) -> bool:
    for point in partial.points:
        for time_sec, freq_hz in trace:
            if abs(point.time - time_sec) <= 0.02 and abs(point.freq - freq_hz) <= radius_hz:
                return True
    return False


def _split_partial(
    partial: Partial,
    trace: list[tuple[float, float]],
    radius_hz: float,
) -> list[Partial]:
    trace_times = [t for t, _ in trace]
    t_min = min(trace_times)
    t_max = max(trace_times)
    remaining = [p for p in partial.points if not (t_min <= p.time <= t_max)]
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
        result.append(Partial(id=str(uuid.uuid4()), points=segment))
    return result
