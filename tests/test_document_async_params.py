import numpy as np
import pytest

from soma.models import AudioInfo
from soma.services.history import HistoryService
from soma.services.playback_service import PlaybackService
from soma.services.preview_service import PreviewService
from soma.session import ProjectSession


class _CaptureComputeManager:
    def __init__(self) -> None:
        self.last_snap_audio: np.ndarray | None = None
        self.last_snap_trace: list[tuple[float, float]] | None = None
        self.last_snap_offset: float | None = None

    def submit_snap(self, _request_id: str, params) -> None:  # type: ignore[no-untyped-def]
        self.last_snap_audio = params.audio
        self.last_snap_trace = params.trace
        self.last_snap_offset = params.time_offset_sec


class _QueueCaptureComputeManager(_CaptureComputeManager):
    def __init__(self) -> None:
        super().__init__()
        self.snap_request_ids: list[str] = []

    def submit_snap(self, request_id: str, params) -> None:  # type: ignore[no-untyped-def]
        super().submit_snap(request_id, params)
        self.snap_request_ids.append(request_id)


def _make_preview_with_audio() -> tuple[ProjectSession, PreviewService, np.ndarray, _CaptureComputeManager]:
    session = ProjectSession()
    playback = PlaybackService(session)
    history = HistoryService(session)
    history.set_callbacks(
        on_settings_applied=lambda: None,
        on_partials_changed=playback.invalidate_cache,
    )
    preview = PreviewService(session, history, on_partials_changed=playback.invalidate_cache)

    audio = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    session.audio_data = audio
    session.audio_info = AudioInfo(
        path="/tmp/test.wav",
        name="test.wav",
        sample_rate=1000,
        duration_sec=1.0,
        channels=1,
        truncated=False,
    )
    manager = _CaptureComputeManager()
    preview._compute_manager = manager  # type: ignore[assignment]
    return session, preview, audio, manager


def test_snap_partial_async_passes_time_roi_audio() -> None:
    _session, preview, audio, manager = _make_preview_with_audio()

    request_id = preview.snap_partial_async([(0.1, 440.0), (0.2, 441.0)])

    assert request_id is not None
    assert manager.last_snap_audio is not None
    assert manager.last_snap_audio is not audio
    assert np.array_equal(manager.last_snap_audio, audio[:450])
    assert manager.last_snap_trace == [(0.1, 440.0), (0.2, 441.0)]
    assert manager.last_snap_offset == 0.0


def test_snap_partial_async_queues_latest_request(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    session = ProjectSession()
    audio = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    session.audio_data = audio
    session.audio_info = AudioInfo(
        path="/tmp/test.wav",
        name="test.wav",
        sample_rate=1000,
        duration_sec=1.0,
        channels=1,
        truncated=False,
    )
    playback = PlaybackService(session)
    history = HistoryService(session)
    history.set_callbacks(
        on_settings_applied=lambda: None,
        on_partials_changed=playback.invalidate_cache,
    )
    preview = PreviewService(session, history, on_partials_changed=playback.invalidate_cache)
    manager = _QueueCaptureComputeManager()
    preview._compute_manager = manager  # type: ignore[assignment]
    monkeypatch.setattr(preview, "_estimate_snap_amp_reference_for_trace", lambda *_args: 1.0)

    first_id = preview.snap_partial_async([(0.1, 440.0), (0.2, 441.0)])
    second_id = preview.snap_partial_async([(0.3, 440.0), (0.4, 441.0)])
    third_id = preview.snap_partial_async([(0.5, 440.0), (0.6, 441.0)])

    with pytest.raises(RuntimeError, match="Snap queue is full"):
        preview.snap_partial_async([(0.7, 440.0), (0.8, 441.0)])

    assert first_id is not None
    assert second_id is not None
    assert third_id is not None
    assert manager.snap_request_ids == [first_id]

    preview._on_snap_result({"type": "snap", "request_id": first_id, "points": [], "error": "Canceled"})
    assert manager.snap_request_ids == [first_id, second_id]
    preview._on_snap_result({"type": "snap", "request_id": second_id, "points": [], "error": "Canceled"})
    assert manager.snap_request_ids == [first_id, second_id, third_id]
