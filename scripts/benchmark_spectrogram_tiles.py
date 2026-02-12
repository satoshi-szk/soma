from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from soma.models import AnalysisSettings
from soma.spectrogram_renderer import SpectrogramRenderer


def main() -> None:
    sample_rate = 44100
    duration_sec = 3600.0
    samples = int(sample_rate * duration_sec)
    t = np.arange(samples, dtype=np.float64) / float(sample_rate)
    audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * t) + 0.1 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)
    settings = AnalysisSettings()
    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        tile_sec = 1.0
        iterations = 20
        warmup = 3
        times_ms: list[float] = []
        for index in range(warmup + iterations):
            start = (index % 10) * (tile_sec * 0.5)
            end = start + tile_sec
            t0 = time.perf_counter()
            renderer.render_tile(
                settings=settings,
                time_start=start,
                time_end=end,
                freq_min=settings.freq_min,
                freq_max=settings.preview_freq_max,
                width=1024,
                height=320,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if index >= warmup:
                times_ms.append(elapsed_ms)
    finally:
        renderer.close()

    arr = np.asarray(times_ms, dtype=np.float64)
    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "tile_sec": tile_sec,
        "iterations": iterations,
        "timings_ms": [round(float(v), 4) for v in times_ms],
        "stats_ms": {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95.0)),
        },
    }
    out_dir = Path("logs/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = out_dir / f"spectrogram-tiles-{stamp}.json"
    latest_file = out_dir / "spectrogram-tiles-latest.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_file)


if __name__ == "__main__":
    main()
