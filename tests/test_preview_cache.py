import time
from pathlib import Path

from soma.models import SpectrogramPreview
from soma.preview_cache import PreviewCacheConfig, build_preview_payload


def test_build_preview_payload_externalizes_jpeg(tmp_path: Path) -> None:
    preview = SpectrogramPreview(
        width=16,
        height=8,
        data=[(index * 7) % 256 for index in range(16 * 8)],
        time_start=0.0,
        time_end=1.0,
        freq_min=20.0,
        freq_max=20000.0,
        duration_sec=2.0,
    )
    cache = PreviewCacheConfig(dir_path=tmp_path, url_prefix="http://127.0.0.1:9999/.soma-cache")

    payload = build_preview_payload(preview, cache, hint="viewport")

    assert payload["width"] == 16
    assert payload["height"] == 8
    assert payload["image_path"].endswith(".jpg")
    assert payload["data"] == []
    filename = payload["image_path"].rsplit("/", 1)[-1]
    file_path = tmp_path / filename
    assert file_path.exists()
    assert file_path.stat().st_size > 0


def test_build_preview_payload_cleans_cache_in_background(tmp_path: Path) -> None:
    preview = SpectrogramPreview(
        width=16,
        height=8,
        data=[(index * 11) % 256 for index in range(16 * 8)],
        time_start=0.0,
        time_end=1.0,
        freq_min=20.0,
        freq_max=20000.0,
        duration_sec=2.0,
    )
    cache = PreviewCacheConfig(dir_path=tmp_path, url_prefix="http://127.0.0.1:9999/.soma-cache", max_entries=3)

    for _ in range(10):
        payload = build_preview_payload(preview, cache, hint="viewport")
        assert payload["image_path"].endswith(".jpg")

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if len(list(tmp_path.glob("*.jpg"))) <= 3:
            break
        time.sleep(0.05)
    assert len(list(tmp_path.glob("*.jpg"))) <= 3
