"""Multiprocess workers for heavy computation (CWT/STFT).

This module provides worker management for offloading heavy spectrogram
computations to separate processes, avoiding GIL contention with the main UI thread.

Invariant: At most 2 worker processes exist at any time (1 viewport + 1 snap).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Empty
from queue import Queue as ThreadQueue
from threading import Lock, Thread
from typing import Any

import numpy as np

from soma.models import AnalysisSettings, PartialPoint, SpectrogramPreview

_logger = logging.getLogger(__name__)

# ワーカープロセスのタイムアウト（5分）
_WORKER_TIMEOUT_SEC = 300.0


@dataclass
class ViewportParams:
    """Parameters for viewport preview computation."""

    audio: np.ndarray
    sample_rate: int
    settings: AnalysisSettings
    time_start: float
    time_end: float
    freq_min: float
    freq_max: float
    width: int
    height: int
    use_stft: bool  # True は STFT のみ（窓幅 > 5 秒）、False は CWT のみ
    stft_amp_reference: float | None = None
    cwt_amp_reference: float | None = None


@dataclass
class SnapParams:
    """Parameters for snap trace computation."""

    audio: np.ndarray
    sample_rate: int
    settings: AnalysisSettings
    trace: list[tuple[float, float]]
    amp_reference: float | None = None
    time_offset_sec: float = 0.0


@dataclass
class ViewportResult:
    """Result from viewport worker."""

    request_id: str
    quality: str  # 'low' または 'high'
    preview: SpectrogramPreview
    amp_reference: float
    final: bool
    error: str | None = None


@dataclass
class SnapResult:
    """Result from snap worker."""

    request_id: str
    points: list[PartialPoint]
    error: str | None = None


def _viewport_worker_fn(
    request_id: str,
    params: ViewportParams,
    result_queue: Queue,  # type: ignore[type-arg]
    log_dir: str | None,
    log_name: str,
) -> None:
    """Viewport computation worker function (runs in separate process)."""
    _configure_worker_logging(log_dir, log_name)
    logger = logging.getLogger(__name__)

    # マルチプロセス時の問題を避けるため、ここで import する。
    from soma.analysis import make_spectrogram, make_spectrogram_stft

    try:
        logger.debug("Viewport worker started: %s", request_id[:8])

        if params.use_stft:
            # STFT 計算（高速・広い時間窓）
            stft_preview, stft_ref = make_spectrogram_stft(
                audio=params.audio,
                sample_rate=params.sample_rate,
                settings=params.settings,
                time_start=params.time_start,
                time_end=params.time_end,
                freq_min=params.freq_min,
                freq_max=params.freq_max,
                width=params.width,
                height=params.height,
                amp_reference=params.stft_amp_reference,
            )

            result_queue.put(
                {
                    "type": "viewport",
                    "request_id": request_id,
                    "quality": "low",
                    "preview": stft_preview.to_dict(),
                    "amp_reference": stft_ref,
                    "final": True,
                    "error": None,
                }
            )
            logger.debug("Viewport worker done (STFT only): %s", request_id[:8])
            return

        # CWT 計算（高品質だが重い）
        logger.debug("Viewport worker starting CWT: %s", request_id[:8])
        cwt_preview, cwt_ref = make_spectrogram(
            audio=params.audio,
            sample_rate=params.sample_rate,
            settings=params.settings,
            time_start=params.time_start,
            time_end=params.time_end,
            freq_min=params.freq_min,
            freq_max=params.freq_max,
            width=params.width,
            height=params.height,
            amp_reference=params.cwt_amp_reference,
        )

        result_queue.put(
            {
                "type": "viewport",
                "request_id": request_id,
                "quality": "high",
                "preview": cwt_preview.to_dict(),
                "amp_reference": cwt_ref,
                "final": True,
                "error": None,
            }
        )
        logger.debug("Viewport worker done (CWT only): %s", request_id[:8])

    except Exception as e:
        logger.exception("Viewport worker error: %s", request_id[:8])
        result_queue.put(
            {
                "type": "viewport",
                "request_id": request_id,
                "quality": "low",
                "preview": None,
                "amp_reference": None,
                "final": True,
                "error": str(e),
            }
        )


def _snap_worker_fn(
    request_id: str,
    params: SnapParams,
    result_queue: Queue,  # type: ignore[type-arg]
    log_dir: str | None,
    log_name: str,
) -> None:
    """Snap computation worker function (runs in separate process)."""
    _configure_worker_logging(log_dir, log_name)
    logger = logging.getLogger(__name__)

    from soma.analysis import snap_trace

    try:
        logger.debug("Snap worker started: %s", request_id[:8])
        metrics: dict[str, float | int] = {}
        points = snap_trace(
            audio=params.audio,
            sample_rate=params.sample_rate,
            settings=params.settings,
            trace=params.trace,
            amp_reference=params.amp_reference,
            metrics=metrics,
        )
        if params.time_offset_sec:
            points = [
                PartialPoint(
                    time=point.time + params.time_offset_sec,
                    freq=point.freq,
                    amp=point.amp,
                )
                for point in points
            ]
        logger.info(
            (
                "Snap worker metrics: id=%s total_ms=%.2f resample_ms=%.2f cwt_ms=%.2f "
                "peak_ms=%.2f cwt_calls=%d tiles=%d tile_sec=%.2f overlap=%.2f "
                "trace_in=%d trace_resampled=%d points_out=%d skipped_windows=%d"
            ),
            request_id[:8],
            float(metrics.get("total_ms", 0.0)),
            float(metrics.get("resample_ms", 0.0)),
            float(metrics.get("cwt_ms", 0.0)),
            float(metrics.get("peak_ms", 0.0)),
            int(metrics.get("cwt_calls", 0)),
            int(metrics.get("tiles", 0)),
            float(metrics.get("tile_duration_sec", 0.0)),
            float(metrics.get("tile_overlap_ratio", 0.0)),
            int(metrics.get("trace_points_in", 0)),
            int(metrics.get("trace_points_resampled", 0)),
            int(metrics.get("snap_points_out", 0)),
            int(metrics.get("skipped_windows", 0)),
        )

        result_queue.put(
            {
                "type": "snap",
                "request_id": request_id,
                "points": [p.to_list() for p in points],
                "error": None,
            }
        )
        logger.debug("Snap worker done: %s", request_id[:8])

    except Exception as e:
        logger.exception("Snap worker error: %s", request_id[:8])
        result_queue.put(
            {
                "type": "snap",
                "request_id": request_id,
                "points": [],
                "error": str(e),
            }
        )


class WorkerBase:
    """Base class for worker management."""

    def __init__(
        self,
        result_callback: Callable[[dict[str, Any]], None],
        log_dir: str | None,
        log_name: str,
    ) -> None:
        self._process: Process | None = None
        self._result_queue: Queue[dict[str, Any]] = mp.Queue()
        self._command_queue: ThreadQueue[dict[str, Any]] = ThreadQueue()
        self._monitor_thread: Thread | None = None
        self._callback = result_callback
        self._current_request_id: str | None = None
        self._start_time: float = 0.0
        self._shutdown = False
        self._log_dir = log_dir
        self._log_name = log_name
        self._state_lock = Lock()
        self._pending_start: tuple[str, Any] | None = None
        self._terminate_deadline: float | None = None
        self._termination_requested = False
        self._termination_reason: str | None = None

        self._start_monitor()

    def request_start(self, request_id: str, params: Any) -> None:
        """Enqueue a start request handled by the monitor thread."""
        self._command_queue.put({"type": "start", "request_id": request_id, "params": params})

    def request_cancel(self) -> None:
        """Enqueue a cancel request handled by the monitor thread."""
        self._command_queue.put({"type": "cancel"})

    def _start_monitor(self) -> None:
        """Start worker management thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._monitor_thread = Thread(
            target=self._monitor_results,
            daemon=True,
            name=f"{self.__class__.__name__}-monitor",
        )
        self._monitor_thread.start()

    def _monitor_results(self) -> None:
        """Manage worker process, results, and lifecycle."""
        _logger.debug("%s monitor thread started", self.__class__.__name__)
        while not self._shutdown:
            self._drain_commands()

            try:
                result = self._result_queue.get(timeout=0.5)
                if result is not None:
                    _logger.debug(
                        "%s received result: type=%s request_id=%s",
                        self.__class__.__name__,
                        result.get("type"),
                        result.get("request_id", "")[:8] if result.get("request_id") else "unknown",
                    )
                    self._callback(result)
            except Empty:
                pass

            self._advance_lifecycle()

        self._shutdown_process()
        _logger.debug("%s monitor thread stopped", self.__class__.__name__)

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._command_queue.get_nowait()
            except Empty:
                return
            command_type = command.get("type")
            if command_type == "start":
                request_id = command["request_id"]
                params = command["params"]
                with self._state_lock:
                    self._pending_start = (request_id, params)
                self._request_terminate(reason="new request")
            elif command_type == "cancel":
                with self._state_lock:
                    self._pending_start = None
                self._request_terminate(reason="cancel")
            elif command_type == "shutdown":
                with self._state_lock:
                    self._pending_start = None
                self._shutdown = True
                self._request_terminate(reason="shutdown")

    def _request_terminate(self, *, reason: str) -> None:
        with self._state_lock:
            process = self._process
        if process is None or not process.is_alive():
            self._try_start_pending()
            return
        _logger.debug(
            "Terminating worker process (%s): %s",
            reason,
            self._current_request_id[:8] if self._current_request_id else "unknown",
        )
        process.terminate()
        with self._state_lock:
            if self._terminate_deadline is None:
                self._terminate_deadline = time.time() + 2.0
            self._termination_requested = True
            self._termination_reason = reason

    def _advance_lifecycle(self) -> None:
        with self._state_lock:
            process = self._process
            current_request_id = self._current_request_id
            start_time = self._start_time
            terminate_deadline = self._terminate_deadline
            termination_requested = self._termination_requested
            termination_reason = self._termination_reason

        if process is None:
            self._try_start_pending()
            return

        if process.is_alive():
            elapsed = time.time() - start_time
            if elapsed > _WORKER_TIMEOUT_SEC:
                _logger.warning(
                    "Worker timeout (%.1fs), terminating: %s",
                    elapsed,
                    current_request_id[:8] if current_request_id else "unknown",
                )
                self._request_terminate(reason="timeout")
                self._callback(
                    {
                        "type": self._worker_type(),
                        "request_id": current_request_id,
                        "error": "Timeout",
                    }
                )
            if terminate_deadline is not None and time.time() > terminate_deadline:
                _logger.warning(
                    "Process did not terminate, killing: %s",
                    current_request_id[:8] if current_request_id else "unknown",
                )
                process.kill()
                with self._state_lock:
                    self._terminate_deadline = time.time() + 0.5
            return

        exitcode = process.exitcode
        process.join(timeout=0.2)
        with self._state_lock:
            self._process = None
            self._terminate_deadline = None
            self._termination_requested = False
            self._termination_reason = None

        if termination_requested:
            _logger.info(
                "Worker terminated (%s, exit code: %s): %s",
                termination_reason or "requested",
                exitcode,
                current_request_id[:8] if current_request_id else "unknown",
            )
        elif exitcode is not None and exitcode != 0:
            _logger.error(
                "Worker crashed (exit code: %d): %s",
                exitcode,
                current_request_id[:8] if current_request_id else "unknown",
            )
        self._try_start_pending()

    def _try_start_pending(self) -> None:
        with self._state_lock:
            pending = self._pending_start
            self._pending_start = None
        if pending is None:
            return
        request_id, params = pending
        self._start_process(request_id, params)

    def _start_process(self, request_id: str, params: Any) -> None:
        with self._state_lock:
            self._current_request_id = request_id
            self._start_time = time.time()
            self._termination_requested = False
            self._termination_reason = None
        try:
            process = self._make_process(request_id, params)
            process.start()
            with self._state_lock:
                self._process = process
            _logger.debug(
                "%s process started: pid=%d",
                self.__class__.__name__,
                process.pid if process else 0,
            )
        except Exception as exc:
            _logger.exception("Failed to start %s process", self.__class__.__name__)
            with self._state_lock:
                self._process = None
            self._callback(self._error_payload(request_id, exc))

    def _shutdown_process(self) -> None:
        with self._state_lock:
            process = self._process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        with self._state_lock:
            self._process = None
            self._terminate_deadline = None

    def _worker_type(self) -> str:
        """Return worker type string for error messages."""
        return "unknown"

    def _make_process(self, request_id: str, params: Any) -> Process:
        raise NotImplementedError

    def _error_payload(self, request_id: str, exc: Exception) -> dict[str, Any]:
        return {"type": self._worker_type(), "request_id": request_id, "error": str(exc)}

    def is_busy(self) -> bool:
        """Check if worker is currently processing."""
        with self._state_lock:
            return self._process is not None and self._process.is_alive()

    def shutdown(self) -> None:
        """Shutdown worker and cleanup."""
        self._command_queue.put({"type": "shutdown"})
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)


class ViewportWorker(WorkerBase):
    """Worker for viewport preview computation."""

    def _worker_type(self) -> str:
        return "viewport"

    def submit(self, request_id: str, params: ViewportParams) -> None:
        """Submit new viewport computation (non-blocking)."""
        _logger.debug("ViewportWorker.submit: %s", request_id[:8])
        self.request_start(request_id, params)

    def _make_process(self, request_id: str, params: Any) -> Process:
        return Process(
            target=_viewport_worker_fn,
            args=(request_id, params, self._result_queue, self._log_dir, self._log_name),
            name="soma-viewport-worker",
        )

    def _error_payload(self, request_id: str, exc: Exception) -> dict[str, Any]:
        return {
            "type": "viewport",
            "request_id": request_id,
            "quality": "low",
            "preview": None,
            "amp_reference": None,
            "final": True,
            "error": f"Failed to start worker: {exc}",
        }


class SnapWorker(WorkerBase):
    """Worker for snap trace computation."""

    def _worker_type(self) -> str:
        return "snap"

    def submit(self, request_id: str, params: SnapParams) -> None:
        """Submit new snap computation (non-blocking)."""
        _logger.debug("SnapWorker.submit: %s", request_id[:8])
        self.request_start(request_id, params)

    def _make_process(self, request_id: str, params: Any) -> Process:
        return Process(
            target=_snap_worker_fn,
            args=(request_id, params, self._result_queue, self._log_dir, self._log_name),
            name="soma-snap-worker",
        )

    def _error_payload(self, request_id: str, exc: Exception) -> dict[str, Any]:
        return {
            "type": "snap",
            "request_id": request_id,
            "points": [],
            "error": f"Failed to start worker: {exc}",
        }


class ComputeManager:
    """Manages viewport and snap workers.

    Ensures the invariant: at most 2 worker processes (1 viewport + 1 snap).
    """

    def __init__(
        self,
        viewport_callback: Callable[[dict[str, Any]], None],
        snap_callback: Callable[[dict[str, Any]], None],
    ) -> None:
        from soma.logging_utils import get_session_log_dir

        log_dir = str(get_session_log_dir("soma"))
        self._viewport_worker = ViewportWorker(viewport_callback, log_dir, "soma-worker-viewport.log")
        self._snap_worker = SnapWorker(snap_callback, log_dir, "soma-worker-snap.log")
        _logger.info("ComputeManager initialized")

    def submit_viewport(self, request_id: str, params: ViewportParams) -> None:
        """Submit viewport preview computation."""
        self._viewport_worker.submit(request_id, params)

    def submit_snap(self, request_id: str, params: SnapParams) -> None:
        """Submit snap trace computation."""
        self._snap_worker.submit(request_id, params)

    def cancel_viewport(self) -> None:
        """Cancel current viewport computation."""
        self._viewport_worker.request_cancel()

    def cancel_snap(self) -> None:
        """Cancel current snap computation."""
        self._snap_worker.request_cancel()

    def cancel_all(self) -> None:
        """Cancel all computations."""
        self._viewport_worker.request_cancel()
        self._snap_worker.request_cancel()

    def is_viewport_busy(self) -> bool:
        """Check if viewport worker is busy."""
        return self._viewport_worker.is_busy()

    def is_snap_busy(self) -> bool:
        """Check if snap worker is busy."""
        return self._snap_worker.is_busy()

    def shutdown(self) -> None:
        """Shutdown all workers."""
        _logger.info("ComputeManager shutting down")
        self._viewport_worker.shutdown()
        self._snap_worker.shutdown()


def _configure_worker_logging(log_dir: str | None, log_name: str) -> None:
    log_level = logging.DEBUG if os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"} else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(log_level)

    if log_dir:
        from logging.handlers import RotatingFileHandler
        from pathlib import Path

        log_path = Path(log_dir) / log_name
        file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        existing_files = {getattr(handler, "baseFilename", None) for handler in root.handlers}
        if str(log_path) not in existing_files:
            root.addHandler(file_handler)

    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
