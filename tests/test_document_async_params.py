import numpy as np
import pytest

from soma.document import SomaDocument
from soma.models import AudioInfo


class _CaptureComputeManager:
    def __init__(self) -> None:
        self.last_snap_audio: np.ndarray | None = None
        self.last_snap_trace: list[tuple[float, float]] | None = None
        self.last_snap_offset: float | None = None
        self.last_viewport_audio: np.ndarray | None = None

    def submit_snap(self, _request_id: str, params) -> None:  # type: ignore[no-untyped-def]
        self.last_snap_audio = params.audio
        self.last_snap_trace = params.trace
        self.last_snap_offset = params.time_offset_sec

    def submit_viewport(self, _request_id: str, params) -> None:  # type: ignore[no-untyped-def]
        self.last_viewport_audio = params.audio


class _QueueCaptureComputeManager(_CaptureComputeManager):
    def __init__(self) -> None:
        super().__init__()
        self.snap_request_ids: list[str] = []

    def submit_snap(self, request_id: str, params) -> None:  # type: ignore[no-untyped-def]
        super().submit_snap(request_id, params)
        self.snap_request_ids.append(request_id)


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


def test_snap_partial_async_passes_time_roi_audio() -> None:
    doc, audio, manager = _make_doc_with_audio()

    request_id = doc.snap_partial_async([(0.1, 440.0), (0.2, 441.0)])

    assert request_id is not None
    assert manager.last_snap_audio is not None
    assert manager.last_snap_audio is not audio
    assert np.array_equal(manager.last_snap_audio, audio[:450])
    assert manager.last_snap_trace == [(0.1, 440.0), (0.2, 441.0)]
    assert manager.last_snap_offset == 0.0


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


def test_snap_partial_async_queues_latest_request(monkeypatch) -> None:  # type: ignore[no-untyped-def]
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
    manager = _QueueCaptureComputeManager()
    doc._compute_manager = manager  # type: ignore[assignment]
    monkeypatch.setattr(doc, "_estimate_snap_amp_reference_for_trace", lambda *_args: 1.0)

    first_id = doc.snap_partial_async([(0.1, 440.0), (0.2, 441.0)])
    second_id = doc.snap_partial_async([(0.3, 440.0), (0.4, 441.0)])
    third_id = doc.snap_partial_async([(0.5, 440.0), (0.6, 441.0)])

    with pytest.raises(RuntimeError, match="Snap queue is full"):
        doc.snap_partial_async([(0.7, 440.0), (0.8, 441.0)])

    assert first_id is not None
    assert second_id is not None
    assert third_id is not None
    assert manager.snap_request_ids == [first_id]

    doc._on_snap_result({"type": "snap", "request_id": first_id, "points": [], "error": "Canceled"})
    assert manager.snap_request_ids == [first_id, second_id]
    doc._on_snap_result({"type": "snap", "request_id": second_id, "points": [], "error": "Canceled"})
    assert manager.snap_request_ids == [first_id, second_id, third_id]
