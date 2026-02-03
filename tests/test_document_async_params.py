import numpy as np

from soma.document import SomaDocument
from soma.models import AudioInfo


class _CaptureComputeManager:
    def __init__(self) -> None:
        self.last_snap_audio: np.ndarray | None = None
        self.last_viewport_audio: np.ndarray | None = None

    def submit_snap(self, _request_id: str, params) -> None:  # type: ignore[no-untyped-def]
        self.last_snap_audio = params.audio

    def submit_viewport(self, _request_id: str, params) -> None:  # type: ignore[no-untyped-def]
        self.last_viewport_audio = params.audio


def _make_doc_with_audio() -> tuple[SomaDocument, np.ndarray, _CaptureComputeManager]:
    doc = SomaDocument()
    audio = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    doc.audio_data = audio
    doc.audio_info = AudioInfo(
        path="/tmp/test.wav",
        name="test.wav",
        sample_rate=1000,
        duration_sec=1.0,
        channels=1,
        truncated=False,
    )
    manager = _CaptureComputeManager()
    doc._compute_manager = manager  # type: ignore[assignment]
    return doc, audio, manager


def test_snap_partial_async_passes_original_audio_reference() -> None:
    doc, audio, manager = _make_doc_with_audio()

    request_id = doc.snap_partial_async([(0.1, 440.0), (0.2, 441.0)])

    assert request_id is not None
    assert manager.last_snap_audio is audio


def test_start_viewport_preview_async_passes_original_audio_reference() -> None:
    doc, audio, manager = _make_doc_with_audio()

    request_id = doc.start_viewport_preview_async(
        time_start=0.0,
        time_end=0.5,
        freq_min=20.0,
        freq_max=500.0,
        width=320,
        height=200,
    )

    assert request_id != ""
    assert manager.last_viewport_audio is audio
