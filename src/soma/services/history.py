from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from soma.models import AnalysisSettings, Partial, SourceInfo
from soma.session import ProjectSession


@dataclass
class ProjectState:
    settings: AnalysisSettings | None = None
    partials: dict[str, Partial | None] | None = None
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


class HistoryService:
    def __init__(
        self,
        session: ProjectSession,
        on_settings_applied: Callable[[], None],
        on_partials_changed: Callable[[], None],
    ) -> None:
        self._session = session
        self._on_settings_applied = on_settings_applied
        self._on_partials_changed = on_partials_changed
        self._manager = UndoRedoManager()

    def clear(self) -> None:
        self._manager.clear()

    def snapshot_state(
        self,
        *,
        partial_ids: Iterable[str] | None = None,
        include_settings: bool = False,
    ) -> ProjectState:
        snapshot: dict[str, Partial | None] | None = None
        if partial_ids is not None:
            snapshot = {}
            for partial_id in partial_ids:
                partial = self._session.store.get(partial_id)
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
            settings=self._session.settings if include_settings else None,
            partials=snapshot,
            source_info=self._session.source_info,
        )

    def record(self, before: ProjectState, after: ProjectState) -> None:
        self._manager.push(HistoryEntry(before=before, after=after))

    def undo(self) -> None:
        entry = self._manager.undo()
        if entry is None:
            return
        self._apply_state(entry.before)
        self._manager.push_redo(entry)

    def redo(self) -> None:
        entry = self._manager.redo()
        if entry is None:
            return
        self._apply_state(entry.after)
        self._manager.push_undo(entry)

    def _apply_state(self, state: ProjectState) -> None:
        if state.settings is not None:
            self._session.settings = state.settings
            if self._session.audio_data is not None and self._session.audio_info is not None:
                self._on_settings_applied()
        if state.partials is None:
            return
        for partial_id, snapshot in state.partials.items():
            if snapshot is None:
                self._session.store.remove(partial_id)
                self._session.synth.remove_partial(partial_id)
                continue
            self._session.store.update(snapshot)
            self._session.synth.apply_partial(snapshot)
        self._on_partials_changed()
