#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from soma.analysis import snap_trace
from soma.models import AnalysisSettings


@dataclass(frozen=True)
class BenchCase:
    name: str
    duration_sec: float
    trace_start_sec: float
    trace_end_sec: float
    base_freq_hz: float
    sweep_hz: float
    vibrato_hz: float
    vibrato_rate_hz: float
    trace_error_hz: float


def _build_cases() -> list[BenchCase]:
    return [
        BenchCase(
            name="short_trace",
            duration_sec=2.0,
            trace_start_sec=0.20,
            trace_end_sec=1.00,
            base_freq_hz=330.0,
            sweep_hz=80.0,
            vibrato_hz=6.0,
            vibrato_rate_hz=5.0,
            trace_error_hz=7.0,
        ),
        BenchCase(
            name="medium_trace",
            duration_sec=8.0,
            trace_start_sec=0.50,
            trace_end_sec=5.00,
            base_freq_hz=440.0,
            sweep_hz=220.0,
            vibrato_hz=12.0,
            vibrato_rate_hz=4.0,
            trace_error_hz=10.0,
        ),
        BenchCase(
            name="long_trace",
            duration_sec=12.0,
            trace_start_sec=1.00,
            trace_end_sec=8.0,
            base_freq_hz=360.0,
            sweep_hz=220.0,
            vibrato_hz=18.0,
            vibrato_rate_hz=3.0,
            trace_error_hz=12.0,
        ),
    ]


def _frequency_curve(time_sec: np.ndarray, case: BenchCase) -> np.ndarray:
    sweep = case.sweep_hz * np.sin(2.0 * np.pi * 0.08 * time_sec)
    vibrato = case.vibrato_hz * np.sin(2.0 * np.pi * case.vibrato_rate_hz * time_sec)
    return np.maximum(30.0, case.base_freq_hz + sweep + vibrato)


def _synthesize_audio(sample_rate: int, case: BenchCase) -> np.ndarray:
    sample_count = int(case.duration_sec * sample_rate)
    t = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    freq = _frequency_curve(t, case)
    phase = 2.0 * np.pi * np.cumsum(freq) / float(sample_rate)
    fundamental = 0.60 * np.sin(phase)
    harmonic = 0.15 * np.sin(2.0 * phase + 0.2)
    tone = (fundamental + harmonic).astype(np.float32)
    return tone


def _build_trace(case: BenchCase, settings: AnalysisSettings) -> list[tuple[float, float]]:
    step_sec = settings.time_resolution_ms / 1000.0
    time_points = np.arange(case.trace_start_sec, case.trace_end_sec + step_sec * 0.5, step_sec, dtype=np.float64)
    true_curve = _frequency_curve(time_points, case)
    guided = true_curve + case.trace_error_hz * np.sin(2.0 * np.pi * 0.6 * time_points)
    return list(zip(time_points.tolist(), guided.tolist(), strict=True))


def _summary_ms(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "mean_ms": float(statistics.fmean(ordered)),
        "p50_ms": float(statistics.median(ordered)),
        "p95_ms": float(ordered[p95_index]),
    }


def run_benchmark(
    sample_rate: int,
    warmup: int,
    iterations: int,
    settings: AnalysisSettings,
    case_names: set[str] | None = None,
    mode: str = "tiled",
) -> dict[str, object]:
    results: dict[str, object] = {
        "captured_at": datetime.now(UTC).isoformat(),
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sample_rate": sample_rate,
            "warmup": warmup,
            "iterations": iterations,
            "mode": mode,
            "settings": asdict(settings),
        },
        "cases": [],
    }

    available_cases = _build_cases()
    if case_names is None:
        selected_cases = available_cases
    else:
        selected_cases = [case for case in available_cases if case.name in case_names]
        if not selected_cases:
            valid = ", ".join(case.name for case in available_cases)
            raise ValueError(f"No matching cases for --cases. Available: {valid}")

    for case in selected_cases:
        audio = _synthesize_audio(sample_rate=sample_rate, case=case)
        trace = _build_trace(case=case, settings=settings)
        timings_ms: list[float] = []
        points_count: list[int] = []

        for index in range(warmup + iterations):
            start = time.perf_counter()
            tile_duration_sec = 0.75 if mode == "tiled" else 0.0
            tile_overlap_ratio = 0.5 if mode == "tiled" else 0.0
            points = snap_trace(
                audio=audio,
                sample_rate=sample_rate,
                settings=settings,
                trace=trace,
                tile_duration_sec=tile_duration_sec,
                tile_overlap_ratio=tile_overlap_ratio,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if index >= warmup:
                timings_ms.append(elapsed_ms)
                points_count.append(len(points))

        if not points_count or min(points_count) <= 0:
            raise RuntimeError(f"Case '{case.name}' produced empty snap points.")

        case_result = {
            "name": case.name,
            "params": asdict(case),
            "trace_points": len(trace),
            "snap_points_min": int(min(points_count)),
            "snap_points_max": int(max(points_count)),
            "timing": _summary_ms(timings_ms),
            "timings_ms": [round(value, 4) for value in timings_ms],
        }
        print(
            f"[{case.name}] trace_points={case_result['trace_points']} "
            f"p50={case_result['timing']['p50_ms']:.2f}ms "
            f"p95={case_result['timing']['p95_ms']:.2f}ms "
            f"mean={case_result['timing']['mean_ms']:.2f}ms"
        )
        results["cases"].append(case_result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark current snap_trace implementation.")
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Comma separated case names (short_trace,medium_trace,long_trace) or 'all'.",
    )
    parser.add_argument(
        "--mode",
        choices=["pointwise", "tiled"],
        default="tiled",
        help="Snap mode to benchmark. Use pointwise for legacy baseline and tiled for optimized path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: logs/benchmarks/snap-baseline-<timestamp>.json",
    )
    args = parser.parse_args()

    settings = AnalysisSettings(time_resolution_ms=10.0)
    case_names: set[str] | None = None
    if args.cases != "all":
        case_names = {item.strip() for item in args.cases.split(",") if item.strip()}

    benchmark = run_benchmark(
        sample_rate=args.sample_rate,
        warmup=args.warmup,
        iterations=args.iterations,
        settings=settings,
        case_names=case_names,
        mode=args.mode,
    )

    output = args.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output = Path("logs") / "benchmarks" / f"snap-{args.mode}-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    print(f"Wrote benchmark report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
