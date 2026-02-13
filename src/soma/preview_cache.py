from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from soma.models import SpectrogramPreview

logger = logging.getLogger(__name__)
_CLEANUP_TRIGGER_MARGIN = 16
_cleanup_states: dict[str, _CleanupState] = {}
_cleanup_states_lock = threading.Lock()


@dataclass(frozen=True)
class PreviewCacheConfig:
    dir_path: Path
    url_prefix: str
    max_entries: int = 200


class _CleanupState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.approx_entries = 0
        self.needs_initial_scan = True
        self.writes_during_run = 0


def build_preview_payload(
    preview: SpectrogramPreview,
    cache: PreviewCacheConfig | None,
    hint: str | None = None,
) -> dict[str, Any]:
    if cache is None:
        return preview.to_dict()
    payload = _write_preview_file(preview, cache, hint)
    if payload is None:
        return preview.to_dict()
    return payload


def _write_preview_file(
    preview: SpectrogramPreview,
    cache: PreviewCacheConfig,
    hint: str | None,
) -> dict[str, Any] | None:
    try:
        cache.dir_path.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        suffix = uuid.uuid4().hex[:8]
        safe_hint = _sanitize_hint(hint)
        filename = f"{safe_hint}-{stamp}-{suffix}.jpg" if safe_hint else f"preview-{stamp}-{suffix}.jpg"
        file_path = cache.dir_path / filename
        file_path.write_bytes(_render_preview_jpeg(preview))
        _notify_cache_write(cache)
        image_path = _join_url(cache.url_prefix, filename)
        return {
            "width": preview.width,
            "height": preview.height,
            "time_start": preview.time_start,
            "time_end": preview.time_end,
            "freq_min": preview.freq_min,
            "freq_max": preview.freq_max,
            "duration_sec": preview.duration_sec,
            "data": [],
            "image_path": image_path,
        }
    except Exception:
        logger.exception("failed to write preview cache")
        return None


def _notify_cache_write(cache: PreviewCacheConfig) -> None:
    state = _cleanup_state(cache)
    should_start = False
    margin = (
        0
        if cache.max_entries <= _CLEANUP_TRIGGER_MARGIN
        else min(_CLEANUP_TRIGGER_MARGIN, cache.max_entries // 10)
    )
    with state.lock:
        state.approx_entries += 1
        if state.running:
            state.writes_during_run += 1
        over_limit = state.approx_entries > (cache.max_entries + margin)
        if state.needs_initial_scan:
            over_limit = True
        if over_limit and not state.running:
            state.running = True
            state.writes_during_run = 0
            should_start = True
    if should_start:
        thread = threading.Thread(
            target=_run_cleanup_worker,
            args=(cache, state),
            name="soma-preview-cleanup",
            daemon=True,
        )
        thread.start()


def _cleanup_state(cache: PreviewCacheConfig) -> _CleanupState:
    key = str(cache.dir_path.resolve())
    with _cleanup_states_lock:
        state = _cleanup_states.get(key)
        if state is None:
            state = _CleanupState()
            _cleanup_states[key] = state
    return state


def _run_cleanup_worker(cache: PreviewCacheConfig, state: _CleanupState) -> None:
    kept_count = _cleanup_cache(cache)
    with state.lock:
        extra_writes = state.writes_during_run
        state.running = False
        state.needs_initial_scan = False
        state.writes_during_run = 0
        state.approx_entries = kept_count + extra_writes


def _cleanup_cache(cache: PreviewCacheConfig) -> int:
    try:
        entries = sorted(
            (path for path in cache.dir_path.glob("*.jpg") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if len(entries) <= cache.max_entries:
            return len(entries)
        for path in entries[cache.max_entries :]:
            try:
                path.unlink()
            except Exception:
                logger.debug("failed to remove cached preview: %s", path, exc_info=True)
        return cache.max_entries
    except Exception:
        logger.debug("failed to cleanup preview cache", exc_info=True)
        return max(0, cache.max_entries)


def _join_url(prefix: str, filename: str) -> str:
    if not prefix:
        return filename
    if prefix.startswith("http://") or prefix.startswith("https://"):
        return f"{prefix.rstrip('/')}/{filename}"
    trimmed = prefix.strip("/")
    if not trimmed:
        return filename
    return f"{trimmed}/{filename}"


def _sanitize_hint(value: str | None) -> str:
    if not value:
        return ""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value)
    cleaned = cleaned.strip("-_")
    return cleaned[:40]


def _render_preview_jpeg(preview: SpectrogramPreview) -> bytes:
    if preview.width <= 0 or preview.height <= 0:
        raise ValueError("invalid preview dimensions")
    expected = preview.width * preview.height
    if len(preview.data) != expected:
        raise ValueError(f"invalid preview data length: expected={expected} got={len(preview.data)}")

    if isinstance(preview.data, bytes):
        levels = np.frombuffer(preview.data, dtype=np.uint8).reshape((preview.height, preview.width))
    else:
        levels = np.asarray(preview.data, dtype=np.uint8).reshape((preview.height, preview.width))
    rgb = _apply_magma_like_colormap(levels)
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return buffer.getvalue()


def _apply_magma_like_colormap(levels: np.ndarray) -> np.ndarray:
    # Frontend の旧カラーマップに合わせる。
    stops = np.asarray(
        [
            [0, 0, 4],
            [50, 18, 91],
            [121, 40, 130],
            [189, 55, 84],
            [249, 142, 8],
            [252, 253, 191],
        ],
        dtype=np.float32,
    )

    t = (levels.astype(np.float32) / 255.0) * float(stops.shape[0] - 1)
    idx = np.floor(t).astype(np.int32)
    idx = np.clip(idx, 0, stops.shape[0] - 1)
    frac = (t - idx.astype(np.float32))[..., None]
    next_idx = np.minimum(idx + 1, stops.shape[0] - 1)
    start = stops[idx]
    end = stops[next_idx]
    rgb = np.rint(start + (end - start) * frac).astype(np.uint8)
    return cast(np.ndarray, rgb)
