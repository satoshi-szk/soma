import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from soma.persistence import load_project, parse_source
from soma.services.history import HistoryService
from soma.services.playback_service import PlaybackService
from soma.services.preview_service import PreviewService
from soma.services.project_service import ProjectService
from soma.session import ProjectSession


def _write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    wavfile.write(path, sample_rate, data)


def test_save_project_bundles_temporary_source(tmp_path: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, prefix="soma-drop-", suffix=".wav") as handle:
        temp_path = Path(handle.name)

    try:
        sample_rate = 44100
        duration_sec = 0.1
        samples = int(sample_rate * duration_sec)
        data = (0.2 * np.sin(2 * np.pi * 440.0 * np.arange(samples) / sample_rate)).astype(np.float32)
        _write_wav(temp_path, sample_rate, data)

        session = ProjectSession()
        playback = PlaybackService(session)
        history = HistoryService(session)
        history.set_callbacks(
            on_settings_applied=lambda: None,
            on_partials_changed=playback.invalidate_cache,
        )
        preview = PreviewService(session, history, on_partials_changed=playback.invalidate_cache)
        project = ProjectService(session, history, playback, preview)

        project.load_audio(temp_path, display_name="original.wav")

        project_path = tmp_path / "project.soma"
        project.save_project(project_path)

        payload = load_project(project_path)
        source = parse_source(payload)
        assert source is not None
        assert source.file_path.startswith("project_assets/")
        assert source.file_path.endswith("original.wav")
        assert (project_path.parent / source.file_path).exists()
    finally:
        temp_path.unlink(missing_ok=True)
