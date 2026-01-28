from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soma.models import SpectrogramPreview

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviewCacheConfig:
    dir_path: Path
    url_prefix: str
    max_entries: int = 200


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
        filename = f"{safe_hint}-{stamp}-{suffix}.bin" if safe_hint else f"preview-{stamp}-{suffix}.bin"
        file_path = cache.dir_path / filename
        file_path.write_bytes(bytes(preview.data))
        _cleanup_cache(cache)
        data_path = _join_url(cache.url_prefix, filename)
        return {
            "width": preview.width,
            "height": preview.height,
            "time_start": preview.time_start,
            "time_end": preview.time_end,
            "freq_min": preview.freq_min,
            "freq_max": preview.freq_max,
            "duration_sec": preview.duration_sec,
            "data_path": data_path,
            "data_length": len(preview.data),
        }
    except Exception:
        logger.exception("failed to write preview cache")
        return None


def _cleanup_cache(cache: PreviewCacheConfig) -> None:
    try:
        entries = sorted(
            (path for path in cache.dir_path.glob("*.bin") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if len(entries) <= cache.max_entries:
            return
        for path in entries[cache.max_entries :]:
            try:
                path.unlink()
            except Exception:
                logger.debug("failed to remove cached preview: %s", path, exc_info=True)
    except Exception:
        logger.debug("failed to cleanup preview cache", exc_info=True)


def _join_url(prefix: str, filename: str) -> str:
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
