from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import webview

from soma.analysis import load_audio, make_spectrogram_preview


class SomaApi:
    def __init__(self) -> None:
        self._last_audio_path: str | None = None

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def open_audio(self) -> dict[str, Any]:
        window = webview.windows[0] if webview.windows else None
        if window is None:
            return {"status": "error", "message": "Window not ready"}

        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=("Audio Files (*.wav;*.WAV)",),
        )
        if not result:
            return {"status": "cancelled"}

        path = Path(result[0])
        try:
            info, audio = load_audio(path)
            preview = make_spectrogram_preview(audio, info.sample_rate)
        except Exception as exc:  # pragma: no cover - surface errors to UI
            return {"status": "error", "message": str(exc)}

        self._last_audio_path = info.path
        return {
            "status": "ok",
            "audio": info.to_dict(),
            "preview": preview.to_dict(),
        }


def resolve_frontend_url() -> str:
    dev_url = os.environ.get("SOMA_DEV_SERVER_URL")
    if dev_url:
        return dev_url

    force_dev = os.environ.get("SOMA_DEV", "").lower() in {"1", "true", "yes"}
    if force_dev:
        return "http://localhost:5173"

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        index_path = Path(bundle_root) / "frontend" / "dist" / "index.html"
        if index_path.exists():
            return index_path.as_uri()

    repo_root = Path(__file__).resolve().parents[2]
    repo_index = repo_root / "frontend" / "dist" / "index.html"
    if repo_index.exists():
        return repo_index.as_uri()

    package_index = Path(__file__).resolve().parent / "ui" / "index.html"
    if package_index.exists():
        return package_index.as_uri()

    return "http://localhost:5173"


def main() -> None:
    url = resolve_frontend_url()
    api = SomaApi()
    webview.create_window(
        "SOMA",
        url=url,
        js_api=api,
        width=1280,
        height=800,
        min_size=(960, 600),
        background_color="#f4f1ed",
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()
