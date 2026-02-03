import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from soma.document import SomaDocument
from soma.persistence import load_project, parse_source


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

        doc = SomaDocument()
        doc.load_audio(temp_path, display_name="original.wav")

        project_path = tmp_path / "project.soma"
        doc.save_project(project_path)

        payload = load_project(project_path)
        source = parse_source(payload)
        assert source is not None
        assert source.file_path.startswith("project_assets/")
        assert source.file_path.endswith("original.wav")
        assert (project_path.parent / source.file_path).exists()
    finally:
        temp_path.unlink(missing_ok=True)
