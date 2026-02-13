from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable

from soma.models import Partial, PartialPoint, generate_bright_color
from soma.services.document_utils import intersects_trace, split_partial
from soma.services.history import HistoryService
from soma.session import ProjectSession


class PartialEditService:
    def __init__(
        self,
        session: ProjectSession,
        history: HistoryService,
        on_partials_changed: Callable[[], None],
    ) -> None:
        self._session = session
        self._history = history
        self._on_partials_changed = on_partials_changed

    def erase_path(self, trace: list[tuple[float, float]], radius_hz: float = 40.0) -> list[Partial]:
        if not trace:
            return []
        affected: list[Partial] = []
        for partial in self._session.store.all():
            if partial.is_muted:
                continue
            if intersects_trace(partial, trace, radius_hz):
                affected.append(partial)
        if not affected:
            return []
        segments: list[Partial] = []
        for partial in affected:
            segments.extend(split_partial(partial, trace, radius_hz))
        removed_ids = [p.id for p in affected]
        added_ids = [segment.id for segment in segments]
        tracked_ids = removed_ids + added_ids
        before = self._history.snapshot_state(partial_ids=tracked_ids)
        for partial in affected:
            self._session.store.remove(partial.id)
            self._session.synth.remove_partial(partial.id)
        for segment in segments:
            self._session.store.add(segment)
            self._session.synth.apply_partial(segment)
        after = self._history.snapshot_state(partial_ids=tracked_ids)
        self._history.record(before, after)
        self._on_partials_changed()
        return self._session.store.all()

    def toggle_mute(self, partial_id: str) -> Partial | None:
        partial = self._session.store.get(partial_id)
        if partial is None:
            return None
        before = self._history.snapshot_state(partial_ids=[partial_id])
        partial.is_muted = not partial.is_muted
        self._session.store.update(partial)
        self._session.synth.apply_partial(partial)
        after = self._history.snapshot_state(partial_ids=[partial_id])
        self._history.record(before, after)
        self._on_partials_changed()
        return partial

    def delete_partials(self, partial_ids: Iterable[str]) -> None:
        ids = list(partial_ids)
        if not ids:
            return
        before = self._history.snapshot_state(partial_ids=ids)
        for partial_id in ids:
            self._session.store.remove(partial_id)
            self._session.synth.remove_partial(partial_id)
        after = self._history.snapshot_state(partial_ids=ids)
        self._history.record(before, after)
        self._on_partials_changed()

    def update_partial_points(self, partial_id: str, points: list[PartialPoint]) -> Partial | None:
        partial = self._session.store.get(partial_id)
        if partial is None:
            return None
        before = self._history.snapshot_state(partial_ids=[partial_id])
        updated = Partial(id=partial_id, points=points, is_muted=partial.is_muted, color=partial.color)
        self._session.store.update(updated)
        self._session.synth.apply_partial(updated)
        after = self._history.snapshot_state(partial_ids=[partial_id])
        self._history.record(before, after)
        self._on_partials_changed()
        return updated

    def merge_partials(self, first_id: str, second_id: str) -> Partial | None:
        first = self._session.store.get(first_id)
        second = self._session.store.get(second_id)
        if first is None or second is None:
            return None
        merged_id = str(uuid.uuid4())
        tracked_ids = [first_id, second_id, merged_id]
        before = self._history.snapshot_state(partial_ids=tracked_ids)
        merged_points = sorted(first.points + second.points, key=lambda p: p.time)
        merged = Partial(id=merged_id, points=merged_points, color=generate_bright_color())
        self._session.store.remove(first_id)
        self._session.store.remove(second_id)
        self._session.synth.remove_partial(first_id)
        self._session.synth.remove_partial(second_id)
        self._session.store.add(merged)
        self._session.synth.apply_partial(merged)
        after = self._history.snapshot_state(partial_ids=tracked_ids)
        self._history.record(before, after)
        self._on_partials_changed()
        return merged

    def hit_test(self, time_sec: float, freq_hz: float, tolerance: float = 0.05) -> str | None:
        return self._session.store.hit_test(
            time_sec=time_sec,
            freq_hz=freq_hz,
            freq_min=self._session.settings.spectrogram.freq_min,
            freq_max=self._session.settings.spectrogram.freq_max,
            tolerance=tolerance,
        )

    def select_in_box(self, time_start: float, time_end: float, freq_start: float, freq_end: float) -> list[str]:
        return self._session.store.select_in_box(time_start, time_end, freq_start, freq_end)
