from __future__ import annotations

import logging
import os
import sys
import threading
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import BinaryIO

_session_log_dir: Path | None = None


def configure_logging(app_name: str = "soma") -> None:
    force_dev = os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"}
    log_dir = get_session_log_dir(app_name, force_dev)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{app_name}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    existing_files = {getattr(handler, "baseFilename", None) for handler in root.handlers}
    if str(log_path) not in existing_files:
        root.addHandler(file_handler)
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        root.addHandler(console_handler)

    logging.captureWarnings(True)
    warnings.simplefilter("default")
    logging.getLogger("pywebview").setLevel(logging.INFO)
    logging.getLogger("pywebview").propagate = True

    def _excepthook(exc_type, exc_value, exc_traceback) -> None:  # type: ignore[no-untyped-def]
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root.exception("uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _excepthook
    root.info("logging initialized at %s", log_path)

    _attach_stdout_stderr(log_path)

_tee_initialized = False


def _attach_stdout_stderr(log_path: Path) -> None:
    global _tee_initialized
    if _tee_initialized:
        return
    log_handle = log_path.open("ab", buffering=0)
    _tee_fd(1, log_handle)
    _tee_fd(2, log_handle)
    _tee_initialized = True


def _tee_fd(target_fd: int, log_handle: BinaryIO) -> None:
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, False)
    os.set_inheritable(write_fd, False)
    original_fd = os.dup(target_fd)
    os.dup2(write_fd, target_fd)
    os.close(write_fd)

    def _reader() -> None:
        with os.fdopen(read_fd, "rb", closefd=True) as reader:
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                os.write(original_fd, chunk)
                log_handle.write(chunk)
                log_handle.flush()

    thread = threading.Thread(target=_reader, name=f"soma-tee-{target_fd}", daemon=True)
    thread.start()


def get_log_dir(app_name: str, force_dev: bool | None = None) -> Path:
    if force_dev is None:
        force_dev = os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"}
    return Path.cwd() / "logs" if force_dev else (Path.home() / f".{app_name}" / "logs")


def get_session_log_dir(app_name: str, force_dev: bool | None = None) -> Path:
    global _session_log_dir
    if _session_log_dir is None:
        base_dir = get_log_dir(app_name, force_dev)
        session_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _session_log_dir = base_dir / session_stamp
    return _session_log_dir
