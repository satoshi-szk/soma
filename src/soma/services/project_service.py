from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from soma.models import AnalysisSettings, AudioInfo, Partial, SourceInfo
from soma.persistence import (
    build_project_payload,
    compute_md5,
    load_project,
    parse_partials,
    parse_playback_settings,
    parse_settings,
    parse_source,
)
from soma.services.document_utils import ensure_soma_extension, sanitize_audio_filename, unique_destination
from soma.services.history import HistoryService
from soma.services.playback_service import PlaybackService
from soma.services.preview_service import PreviewService
from soma.session import ProjectSession


class ProjectService:
    def __init__(
        self,
        session: ProjectSession,
        history: HistoryService,
        playback: PlaybackService,
        preview: PreviewService,
    ) -> None:
        self._session = session
        self._history = history
        self._playback = playback
        self._preview = preview

    def new_project(self) -> None:
        self._preview.cancel_all()
        self._history.clear()
        self._session.reset_for_new_project()
        self._playback.invalidate_cache()

    def load_audio(
        self,
        path: Path,
        max_duration_sec: float | None = None,
        display_name: str | None = None,
    ) -> AudioInfo:
        from soma.analysis import load_audio

        info, audio = load_audio(path, max_duration_sec=max_duration_sec, display_name=display_name)
        source_info = SourceInfo(
            file_path=info.path,
            sample_rate=info.sample_rate,
            duration_sec=info.duration_sec,
            md5_hash=compute_md5(path),
        )
        self._session.apply_loaded_audio(
            info=info,
            audio=audio,
            source_info=source_info,
            initial_mix_buffer=self._playback.mix_buffer(0.5),
        )
        from soma.spectrogram_renderer import SpectrogramRenderer

        self._session._spectrogram_renderer = SpectrogramRenderer(audio=audio, sample_rate=info.sample_rate)
        self._playback.invalidate_cache()
        return info

    def set_settings(self, settings: AnalysisSettings) -> None:
        before = self._history.snapshot_state(include_settings=True)
        self._session.settings = settings
        self._session._snap_amp_reference = None
        after = self._history.snapshot_state(include_settings=True)
        self._history.record(before, after)

    def save_project(self, path: Path) -> None:
        if self._session.source_info is None or self._session.audio_info is None:
            raise ValueError("No audio loaded")
        path = ensure_soma_extension(path)
        source_info = self._prepare_source_info_for_save(path)
        payload = build_project_payload(
            source_info,
            self._session.settings,
            self._playback.playback_settings(),
            self._session.store.all(),
        )
        from soma.persistence import save_project

        save_project(path, payload)
        self._session.project_path = path

    def load_project(self, path: Path) -> dict[str, Any]:
        data = load_project(path)
        source = parse_source(data)
        settings = parse_settings(data)
        playback_settings = parse_playback_settings(data)
        partials = parse_partials(data)
        self._session.settings = settings
        self._playback.set_master_volume(playback_settings.master_volume)
        self._playback.update_playback_settings(playback_settings)
        self._session.store = type(self._session.store)()
        for partial in partials:
            self._session.store.add(partial)
        self._session.project_path = path
        self._session.source_info = source
        self._history.clear()
        return {"source": source, "settings": settings, "playback_settings": playback_settings, "partials": partials}

    def _prepare_source_info_for_save(self, project_path: Path) -> SourceInfo:
        if self._session.source_info is None:
            raise ValueError("No source audio loaded")
        source_info = self._session.source_info
        source_path = Path(source_info.file_path).expanduser()
        if not source_path.is_absolute() and self._session.project_path is not None:
            source_path = (self._session.project_path.parent / source_path).resolve()

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
        self._session.source_info = source_info
        return source_info

    def _bundle_source_audio(self, project_path: Path, source_path: Path, source_info: SourceInfo) -> SourceInfo:
        bundle_dir = project_path.parent / f"{project_path.stem}_assets"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        preferred_name = self._session.audio_info.name if self._session.audio_info is not None else source_path.name
        sanitized = sanitize_audio_filename(preferred_name, fallback=source_path.name)
        destination = unique_destination(bundle_dir / sanitized)
        shutil.copy2(source_path, destination)
        relative_path = destination.relative_to(project_path.parent).as_posix()
        return SourceInfo(
            file_path=relative_path,
            sample_rate=source_info.sample_rate,
            duration_sec=source_info.duration_sec,
            md5_hash=source_info.md5_hash,
        )

    def project_path(self) -> Path | None:
        return self._session.project_path

    def audio_info(self) -> AudioInfo | None:
        return self._session.audio_info

    def settings(self) -> AnalysisSettings:
        return self._session.settings

    def partials(self) -> list[Partial]:
        return self._session.store.all()
