from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from soma.analysis import _apply_preview_tone, _normalize_magnitude_db
from soma.models import SpectrogramSettings
from soma.spectrogram_renderer import SpectrogramRenderer


@dataclass(frozen=True)
class ResolutionProfile:
    key: str
    label: str
    nperseg: int
    hop_samples: int
    nfft: int


PROFILES: tuple[ResolutionProfile, ...] = (
    ResolutionProfile(key="low", label="Low", nperseg=8192, hop_samples=512, nfft=4096),
    ResolutionProfile(key="mid", label="Mid", nperseg=2048, hop_samples=128, nfft=4096),
    ResolutionProfile(key="high", label="High", nperseg=512, hop_samples=32, nfft=4096),
)
NO_CROSSFADE_SPLIT_HZ: tuple[float, float] = (200.0, 2000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "マルチレゾリューションSTFTの各解像度(low/mid/high)を単体レンダリングし、"
            "見た目と生成時間を比較します。"
        ),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("test_projects/We'll Take It (Short).wav"),
        help="入力音源パス",
    )
    parser.add_argument("--time-start", type=float, default=0.0, help="表示開始時刻 (sec)")
    parser.add_argument("--time-span", type=float, default=2.0, help="表示時間幅 (sec)")
    parser.add_argument("--freq-min", type=float, default=20.0, help="最低周波数 (Hz)")
    parser.add_argument("--freq-max", type=float, default=12000.0, help="最高周波数 (Hz)")
    parser.add_argument("--width", type=int, default=1200, help="描画幅 (px)")
    parser.add_argument("--height", type=int, default=320, help="描画高さ (px)")
    parser.add_argument("--warmup", type=int, default=1, help="ウォームアップ回数")
    parser.add_argument("--iterations", type=int, default=5, help="計測反復回数")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/spectrogram-tuning"),
        help="出力ディレクトリ",
    )
    return parser.parse_args()


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1, dtype=np.float32)
    return mono, int(sample_rate)


def _render_single_resolution(
    *,
    renderer: SpectrogramRenderer,
    settings: SpectrogramSettings,
    profile: ResolutionProfile,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    band_image, _target_log_freqs = _compute_band_matrix(
        renderer=renderer,
        settings=settings,
        profile=profile,
        time_start=time_start,
        time_end=time_end,
        freq_min=freq_min,
        freq_max=freq_max,
        width=width,
        height=height,
    )
    image = _matrix_to_image(matrix=band_image, settings=settings)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return image, elapsed_ms


def _compute_band_matrix(
    *,
    renderer: SpectrogramRenderer,
    settings: SpectrogramSettings,
    profile: ResolutionProfile,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = float(renderer._sample_rate)
    total_duration = renderer._length / sample_rate
    start = max(0.0, min(time_start, total_duration))
    end = max(start + 1e-6, min(time_end, total_duration))

    margin_sec = max((profile.nperseg / sample_rate) * 0.5, (end - start) * 0.05, 0.01)
    ext_start = max(0.0, start - margin_sec)
    ext_end = min(total_duration, end + margin_sec)
    ext_start_idx = int(np.floor(ext_start * sample_rate))
    ext_end_idx = int(np.ceil(ext_end * sample_rate))
    ext_end_idx = max(ext_end_idx, ext_start_idx + 1)
    local_audio = np.asarray(renderer._audio[ext_start_idx:ext_end_idx], dtype=np.float32)

    nyquist = sample_rate * 0.5
    effective_min = max(freq_min, settings.freq_min, 20.0)
    effective_max = min(freq_max, settings.preview_freq_max, settings.freq_max, nyquist)
    effective_max = max(effective_max, effective_min * 1.001)

    target_log_freqs = np.geomspace(effective_min, effective_max, height).astype(np.float64)
    view_samples = max((end - start) * sample_rate, 1.0)
    band_cols = renderer._compute_local_band_cols(
        view_samples=view_samples,
        width=width,
        nperseg=profile.nperseg,
        hop_samples=profile.hop_samples,
    )
    if band_cols <= 1:
        center_times = np.array([0.5 * (start + end)], dtype=np.float64)
    else:
        center_times = np.linspace(start, end, band_cols, dtype=np.float64)
    centers = np.rint(center_times * sample_rate).astype(np.int64) - np.int64(ext_start_idx)

    band_mag = renderer._sparse_stft_magnitude(
        local_audio,
        centers=centers,
        nperseg=profile.nperseg,
        nfft=profile.nfft,
    )
    if band_mag.size == 0:
        return np.zeros((height, width), dtype=np.float32), target_log_freqs

    fft_freqs = np.fft.rfftfreq(band_mag.shape[0] * 2 - 2, d=1.0 / sample_rate).astype(np.float64)
    band_image = renderer._interpolate_freq_matrix(
        source_freqs=fft_freqs,
        source_matrix=band_mag,
        target_freqs=target_log_freqs,
    )
    if band_cols != width:
        band_image = renderer._resample_time_matrix(
            source=band_image,
            width=width,
            relative_start_col=0.0,
            relative_end_col=float(max(0, band_cols - 1)),
        )
    return band_image, target_log_freqs


def _matrix_to_image(*, matrix: np.ndarray, settings: SpectrogramSettings) -> np.ndarray:
    # 単体バンドの見た目確認が目的なので、各バンド自身の最大値で正規化する。
    band_reference = float(np.max(matrix)) if matrix.size else 1.0
    if band_reference <= 0.0:
        band_reference = 1.0
    normalized = _normalize_magnitude_db(matrix, reference_max=band_reference)
    normalized = np.flipud(normalized)
    normalized = _apply_preview_tone(normalized, settings)
    image = (normalized * 255).astype(np.uint8, copy=False)
    return image


def _render_no_crossfade_composite(
    *,
    renderer: SpectrogramRenderer,
    settings: SpectrogramSettings,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    matrices: dict[str, np.ndarray] = {}
    target_log_freqs: np.ndarray | None = None
    for profile in PROFILES:
        matrix, freqs = _compute_band_matrix(
            renderer=renderer,
            settings=settings,
            profile=profile,
            time_start=time_start,
            time_end=time_end,
            freq_min=freq_min,
            freq_max=freq_max,
            width=width,
            height=height,
        )
        matrices[profile.key] = matrix
        target_log_freqs = freqs

    if target_log_freqs is None:
        composed = np.zeros((height, width), dtype=np.float32)
    else:
        edge1, edge2 = NO_CROSSFADE_SPLIT_HZ
        low_mask = target_log_freqs < edge1
        mid_mask = (target_log_freqs >= edge1) & (target_log_freqs < edge2)
        high_mask = target_log_freqs >= edge2
        composed = np.zeros((height, width), dtype=np.float32)
        composed[low_mask, :] = matrices["low"][low_mask, :]
        composed[mid_mask, :] = matrices["mid"][mid_mask, :]
        composed[high_mask, :] = matrices["high"][high_mask, :]

    image = _matrix_to_image(matrix=composed, settings=settings)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return image, elapsed_ms


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95.0)),
    }


def _plot_summary(
    *,
    images: dict[str, np.ndarray],
    stats: dict[str, dict[str, float]],
    profiles: tuple[ResolutionProfile, ...],
    start: float,
    end: float,
    freq_min: float,
    freq_max: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
    plot_order = ["low", "mid", "high", "no_crossfade_composite"]
    title_map = {
        "low": "LOW",
        "mid": "MID",
        "high": "HIGH",
        "no_crossfade_composite": "COMPOSITE (No Crossfade)",
    }
    window_map = {p.key: p.nperseg for p in profiles}
    for index, key in enumerate(plot_order):
        row, col = divmod(index, 2)
        if index >= 2:
            col = index - 2
            row = 1
        ax = axes[row, col]
        image = images[key]
        time_edges = np.linspace(start, end, image.shape[1] + 1, dtype=np.float64)
        freq_edges = np.geomspace(freq_min, freq_max, image.shape[0] + 1, dtype=np.float64)
        # 画像は高周波が上になるように反転済みなので、描画時に戻して軸と整合させる。
        ax.pcolormesh(
            time_edges,
            freq_edges,
            np.flipud(image),
            cmap="magma",
            shading="auto",
            vmin=0,
            vmax=255,
        )
        ax.set_yscale("log")
        if key in window_map:
            ax.set_title(f"{title_map[key]} [win={window_map[key]}] ({stats[key]['mean_ms']:.1f} ms)")
        else:
            ax.set_title(f"{title_map[key]} ({stats[key]['mean_ms']:.1f} ms)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    ax_bar = axes[0, 2]
    labels = [title_map[key] for key in plot_order]
    means = [stats[key]["mean_ms"] for key in plot_order]
    p95s = [stats[key]["p95_ms"] for key in plot_order]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.35
    ax_bar.bar(x - width / 2.0, means, width=width, label="mean (ms)")
    ax_bar.bar(x + width / 2.0, p95s, width=width, label="p95 (ms)")
    ax_bar.set_xticks(x, labels)
    ax_bar.set_ylabel("Render Time (ms)")
    ax_bar.set_title("Render Time Comparison")
    ax_bar.legend()
    axes[1, 2].axis("off")

    fig.suptitle("Single-Resolution STFT + Composite Without Crossfade")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    audio, sample_rate = _load_audio(args.audio)
    duration_sec = audio.shape[0] / float(sample_rate)

    time_start = float(max(0.0, args.time_start))
    time_end = min(duration_sec, time_start + max(args.time_span, 1e-6))
    settings = SpectrogramSettings(
        freq_min=float(args.freq_min),
        freq_max=float(args.freq_max),
        preview_freq_max=float(args.freq_max),
    )

    images: dict[str, np.ndarray] = {}
    timings: dict[str, list[float]] = {}
    renderer = SpectrogramRenderer(audio=audio, sample_rate=sample_rate)
    try:
        for profile in PROFILES:
            for _ in range(max(0, args.warmup)):
                _render_single_resolution(
                    renderer=renderer,
                    settings=settings,
                    profile=profile,
                    time_start=time_start,
                    time_end=time_end,
                    freq_min=args.freq_min,
                    freq_max=args.freq_max,
                    width=args.width,
                    height=args.height,
                )

            elapsed_list: list[float] = []
            latest_image = np.zeros((args.height, args.width), dtype=np.uint8)
            for _ in range(max(1, args.iterations)):
                latest_image, elapsed_ms = _render_single_resolution(
                    renderer=renderer,
                    settings=settings,
                    profile=profile,
                    time_start=time_start,
                    time_end=time_end,
                    freq_min=args.freq_min,
                    freq_max=args.freq_max,
                    width=args.width,
                    height=args.height,
                )
                elapsed_list.append(elapsed_ms)

            images[profile.key] = latest_image
            timings[profile.key] = elapsed_list

        for _ in range(max(0, args.warmup)):
            _render_no_crossfade_composite(
                renderer=renderer,
                settings=settings,
                time_start=time_start,
                time_end=time_end,
                freq_min=args.freq_min,
                freq_max=args.freq_max,
                width=args.width,
                height=args.height,
            )
        no_crossfade_timings: list[float] = []
        no_crossfade_image = np.zeros((args.height, args.width), dtype=np.uint8)
        for _ in range(max(1, args.iterations)):
            no_crossfade_image, elapsed_ms = _render_no_crossfade_composite(
                renderer=renderer,
                settings=settings,
                time_start=time_start,
                time_end=time_end,
                freq_min=args.freq_min,
                freq_max=args.freq_max,
                width=args.width,
                height=args.height,
            )
            no_crossfade_timings.append(elapsed_ms)
        images["no_crossfade_composite"] = no_crossfade_image
        timings["no_crossfade_composite"] = no_crossfade_timings
    finally:
        renderer.close()

    stats = {key: _stats(values) for key, values in timings.items()}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"multires-single-resolution-{stamp}.png"
    json_path = args.out_dir / f"multires-single-resolution-{stamp}.json"
    latest_png = args.out_dir / "multires-single-resolution-latest.png"
    latest_json = args.out_dir / "multires-single-resolution-latest.json"

    _plot_summary(
        images=images,
        stats=stats,
        profiles=PROFILES,
        start=time_start,
        end=time_end,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        output_path=png_path,
    )
    latest_png.write_bytes(png_path.read_bytes())

    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "audio_path": str(args.audio),
        "sample_rate": sample_rate,
        "audio_duration_sec": duration_sec,
        "window": {
            "time_start_sec": time_start,
            "time_end_sec": time_end,
            "freq_min_hz": args.freq_min,
            "freq_max_hz": args.freq_max,
            "width_px": args.width,
            "height_px": args.height,
        },
        "timings_ms": timings,
        "stats_ms": stats,
        "profiles": [
            {
                "key": p.key,
                "label": p.label,
                "nperseg": p.nperseg,
                "hop_samples": p.hop_samples,
                "nfft": p.nfft,
            }
            for p in PROFILES
        ]
        + [{"key": "no_crossfade_composite", "label": "Composite (No Crossfade)", "split_hz": NO_CROSSFADE_SPLIT_HZ}],
        "image_path": str(png_path),
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")

    print(png_path)
    print(json_path)


if __name__ == "__main__":
    main()
