from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter
from ssqueezepy import ssq_cwt, ssq_stft

from soma.analysis import _apply_preview_tone, _normalize_magnitude_db
from soma.models import AnalysisSettings, SpectrogramPreview
from soma.spectrogram_renderer import SpectrogramRenderer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Res STFT / Reassigned STFT / CWT の見た目と計算時間を比較する。",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("test_projects/We'll Take It (Short).wav"),
        help="入力音源",
    )
    parser.add_argument("--time-start", type=float, default=0.0, help="表示開始時刻 (sec)")
    parser.add_argument("--time-span", type=float, default=2.0, help="表示時間幅 (sec)")
    parser.add_argument("--freq-min", type=float, default=20.0, help="最低周波数 (Hz)")
    parser.add_argument("--freq-max", type=float, default=12000.0, help="最高周波数 (Hz)")
    parser.add_argument("--width", type=int, default=1600, help="出力幅 (px)")
    parser.add_argument("--height", type=int, default=420, help="出力高さ (px)")
    parser.add_argument("--warmup", type=int, default=1, help="ウォームアップ回数")
    parser.add_argument("--iterations", type=int, default=4, help="計測反復回数")
    parser.add_argument("--reassigned-sigma-time", type=float, default=1.0, help="Reassigned STFT 時間方向ぼかし")
    parser.add_argument("--reassigned-sigma-freq", type=float, default=1.4, help="Reassigned STFT 周波数方向ぼかし")
    parser.add_argument("--fsst-n-fft", type=int, default=4096, help="FSST の n_fft（0で自動）")
    parser.add_argument("--fsst-hop", type=int, default=0, help="FSST の hop_len（0で自動）")
    parser.add_argument("--wsst-nv", type=int, default=32, help="WSST/CWT の voices per octave")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/spectrogram-comparison"),
        help="出力先ディレクトリ",
    )
    return parser.parse_args()


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1, dtype=np.float32), int(sample_rate)


def _to_matrix(preview: SpectrogramPreview) -> np.ndarray:
    if isinstance(preview.data, bytes):
        return np.frombuffer(preview.data, dtype=np.uint8).reshape((preview.height, preview.width))
    return np.asarray(preview.data, dtype=np.uint8).reshape((preview.height, preview.width))


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95.0)),
    }


def _normalize_to_image(matrix: np.ndarray, settings: AnalysisSettings) -> np.ndarray:
    matrix = np.maximum(matrix, 0.0)
    ref = float(np.max(matrix)) if matrix.size else 1.0
    if ref <= 0.0:
        ref = 1.0
    normalized = _normalize_magnitude_db(matrix, reference_max=ref)
    normalized = np.flipud(normalized)
    normalized = _apply_preview_tone(normalized, settings.spectrogram)
    return (normalized * 255).astype(np.uint8, copy=False)


def _resample_time(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.shape[1] == width:
        return matrix.astype(np.float32, copy=False)
    if matrix.shape[1] <= 1:
        return np.repeat(matrix.astype(np.float32), width, axis=1)
    src_x = np.linspace(0.0, 1.0, matrix.shape[1], dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    out = np.empty((matrix.shape[0], width), dtype=np.float32)
    for row in range(matrix.shape[0]):
        out[row] = np.interp(dst_x, src_x, matrix[row], left=0.0, right=0.0).astype(np.float32)
    return out


def _project_to_logfreq_image(
    matrix: np.ndarray,
    source_freqs: np.ndarray,
    settings: AnalysisSettings,
    *,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    source = np.asarray(matrix, dtype=np.float32)
    freqs = np.asarray(source_freqs, dtype=np.float64)
    if freqs.size != source.shape[0]:
        return np.zeros((height, width), dtype=np.uint8)
    if np.all(np.diff(freqs) < 0):
        freqs = freqs[::-1]
        source = source[::-1, :]
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if np.any(freq_mask):
        freqs = freqs[freq_mask]
        source = source[freq_mask, :]
    if freqs.size < 2:
        return np.zeros((height, width), dtype=np.uint8)
    source = _resample_time(source, width=width)
    target_freqs = np.geomspace(freq_min, freq_max, height).astype(np.float64)
    out = np.empty((height, width), dtype=np.float32)
    for col in range(width):
        out[:, col] = np.interp(target_freqs, freqs, source[:, col], left=0.0, right=0.0).astype(np.float32)
    return _normalize_to_image(out, settings)


def _render_multires_stft(
    renderer: SpectrogramRenderer,
    settings: AnalysisSettings,
    *,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, str]:
    # 計測比較ではローカルタイルキャッシュのヒットを避ける。
    with renderer._local_cache_lock:
        renderer._local_cache.clear()
    preview, _ref, quality = renderer.render_tile(
        settings=settings.spectrogram,
        time_start=time_start,
        time_end=time_end,
        freq_min=freq_min,
        freq_max=freq_max,
        width=width,
        height=height,
    )
    return _to_matrix(preview), quality


def _extract_segment(
    audio: np.ndarray,
    sample_rate: int,
    *,
    time_start: float,
    time_end: float,
) -> np.ndarray:
    start_sample = int(max(0.0, time_start) * sample_rate)
    end_sample = int(min(time_end, audio.shape[0] / float(sample_rate)) * sample_rate)
    return np.asarray(audio[start_sample:end_sample], dtype=np.float32)


def _render_reassigned_stft(
    audio: np.ndarray,
    sample_rate: int,
    renderer: SpectrogramRenderer,
    settings: AnalysisSettings,
    *,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
    sigma_time: float,
    sigma_freq: float,
) -> tuple[np.ndarray, dict[str, float]]:
    start_sample = int(max(0.0, time_start) * sample_rate)
    end_sample = int(min(time_end, audio.shape[0] / float(sample_rate)) * sample_rate)
    segment = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    if segment.size < 32:
        return np.zeros((height, width), dtype=np.uint8), {"n_fft": 0.0, "hop_length": 0.0, "win_length": 0.0}

    freq_edges = np.geomspace(freq_min, freq_max, height + 1).astype(np.float64)
    freq_centers = np.sqrt(freq_edges[:-1] * freq_edges[1:])
    time_edges = np.linspace(time_start, time_end, width + 1, dtype=np.float64)
    composed_map = np.zeros((height, width), dtype=np.float32)
    band_specs = renderer._multires_band_specs(float(freq_max))
    band_weights = renderer._band_weights(
        target_freqs=freq_centers,
        band_freq_max=float(freq_max),
        blend_octaves=settings.spectrogram.multires_blend_octaves,
    )

    last_n_fft = 0
    last_hop = 0
    for band_index, (_band_lo, _band_hi, nperseg, _nfft, hop_samples) in enumerate(band_specs):
        n_fft = max(256, int(nperseg))
        if n_fft & (n_fft - 1) != 0:
            n_fft = 1 << int(np.floor(np.log2(n_fft)))
            n_fft = max(256, n_fft)
        hop_length = max(8, int(hop_samples))
        last_n_fft = n_fft
        last_hop = hop_length

        freqs, times, d_reassigned = librosa.reassigned_spectrogram(
            y=segment,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
            fill_nan=True,
            reassign_frequencies=True,
            reassign_times=True,
        )
        mags = np.abs(d_reassigned).astype(np.float32)
        frames = mags.shape[1] if mags.ndim == 2 else 0
        if frames == 0:
            continue
        band_map = np.zeros((height, width), dtype=np.float32)
        for col in range(frames):
            col_freqs = freqs[:, col]
            col_times = times[:, col] + time_start
            col_mags = mags[:, col]
            valid = (
                np.isfinite(col_freqs)
                & np.isfinite(col_times)
                & np.isfinite(col_mags)
                & (col_freqs >= freq_min)
                & (col_freqs <= freq_max)
                & (col_times >= time_start)
                & (col_times <= time_end)
            )
            if not np.any(valid):
                continue
            row_idx = np.searchsorted(freq_edges, col_freqs[valid], side="right") - 1
            col_idx = np.searchsorted(time_edges, col_times[valid], side="right") - 1
            row_idx = np.clip(row_idx, 0, height - 1)
            col_idx = np.clip(col_idx, 0, width - 1)
            np.add.at(band_map, (row_idx, col_idx), col_mags[valid].astype(np.float32))

        composed_map += band_map * band_weights[band_index][:, None]

    smoothed = gaussian_filter(
        composed_map,
        sigma=(max(0.0, float(sigma_freq)), max(0.0, float(sigma_time))),
        mode="nearest",
    ).astype(np.float32)
    return _normalize_to_image(smoothed, settings), {
        "n_fft": float(last_n_fft),
        "hop_length": float(last_hop),
        "win_length": float(last_n_fft),
    }


def _render_fsst(
    audio: np.ndarray,
    sample_rate: int,
    *,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
    settings: AnalysisSettings,
    n_fft: int,
    hop_len: int,
) -> np.ndarray:
    segment = _extract_segment(audio, sample_rate, time_start=time_start, time_end=time_end)
    if segment.size < 32:
        return np.zeros((height, width), dtype=np.uint8)
    n_fft = int(min(max(256, n_fft), max(256, segment.size)))
    if n_fft & (n_fft - 1) != 0:
        n_fft = 1 << int(np.floor(np.log2(n_fft)))
        n_fft = max(256, n_fft)
    if hop_len <= 0:
        # 表示幅と窓長を両立し、過剰疎密を避ける。
        target = max(1, int(np.round(segment.size / max(1, width))))
        hop = int(np.clip(target, 16, max(16, n_fft // 4)))
    else:
        hop = int(np.clip(hop_len, 8, max(8, n_fft // 2)))
    tx, _sx, ssq_freqs, _sfs = ssq_stft(
        segment,
        fs=sample_rate,
        n_fft=n_fft,
        win_len=n_fft,
        hop_len=hop,
        astensor=False,
    )
    tx_mag = gaussian_filter(np.abs(tx).astype(np.float32), sigma=(1.1, 1.8), mode="nearest")
    return _project_to_logfreq_image(
        tx_mag,
        ssq_freqs,
        settings,
        freq_min=freq_min,
        freq_max=freq_max,
        width=width,
        height=height,
    )


def _render_ssqueezepy_cwt(
    audio: np.ndarray,
    sample_rate: int,
    *,
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    width: int,
    height: int,
    settings: AnalysisSettings,
    nv: int,
    use_sst: bool,
    order: int | tuple[int, ...] = 0,
) -> np.ndarray:
    segment = _extract_segment(audio, sample_rate, time_start=time_start, time_end=time_end)
    if segment.size < 32:
        return np.zeros((height, width), dtype=np.uint8)
    tx, wx, ssq_freqs, _scales = ssq_cwt(
        segment,
        fs=sample_rate,
        scales="log",
        ssq_freqs="log",
        maprange=(float(freq_min), float(freq_max)),
        nv=int(max(8, nv)),
        order=order,
        astensor=False,
    )
    base = tx if use_sst else wx
    tx_mag = gaussian_filter(np.abs(base).astype(np.float32), sigma=(1.0, 1.6), mode="nearest")
    return _project_to_logfreq_image(
        tx_mag,
        ssq_freqs,
        settings,
        freq_min=freq_min,
        freq_max=freq_max,
        width=width,
        height=height,
    )


def _run_timed(
    func: Any,
    *,
    warmup: int,
    iterations: int,
) -> tuple[Any, list[float]]:
    for _ in range(max(0, warmup)):
        _ = func(0)
    latest = None
    values: list[float] = []
    for iteration in range(max(1, iterations)):
        t0 = time.perf_counter()
        latest = func(iteration)
        values.append((time.perf_counter() - t0) * 1000.0)
    return latest, values


def _plot(
    *,
    images: dict[str, np.ndarray],
    stats: dict[str, dict[str, float]],
    labels: dict[str, str],
    order: list[str],
    time_start: float,
    time_end: float,
    freq_min: float,
    freq_max: float,
    out_path: Path,
) -> None:
    n_methods = len(order)
    n_panels = n_methods + 1  # + timing bar
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.2 * nrows), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)
    flat_axes = [axes_arr[r, c] for r in range(axes_arr.shape[0]) for c in range(axes_arr.shape[1])]
    for idx, key in enumerate(order):
        ax = flat_axes[idx]
        r, c = divmod(idx, 2)
        image = images[key]
        time_edges = np.linspace(time_start, time_end, image.shape[1] + 1, dtype=np.float64)
        freq_edges = np.geomspace(freq_min, freq_max, image.shape[0] + 1, dtype=np.float64)
        ax.pcolormesh(
            time_edges,
            freq_edges,
            np.flipud(image),
            shading="auto",
            cmap="magma",
            vmin=0,
            vmax=255,
        )
        ax.set_yscale("log")
        ax.set_title(f"{labels[key]} ({stats[key]['mean_ms']:.1f} ms)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    ax_bar = flat_axes[n_methods]
    xs = np.arange(len(order), dtype=np.float64)
    means = [stats[key]["mean_ms"] for key in order]
    p95s = [stats[key]["p95_ms"] for key in order]
    w = 0.38
    ax_bar.bar(xs - w / 2.0, means, width=w, label="mean (ms)")
    ax_bar.bar(xs + w / 2.0, p95s, width=w, label="p95 (ms)")
    ax_bar.set_xticks(xs, [labels[k] for k in order])
    ax_bar.set_ylabel("Render Time (ms)")
    ax_bar.set_title("Timing Comparison")
    ax_bar.legend()
    for ax in flat_axes[n_methods + 1 :]:
        ax.axis("off")

    fig.suptitle("Spectrogram Method Comparison (Large View)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    audio, sample_rate = _load_audio(args.audio)
    duration = audio.shape[0] / float(sample_rate) if sample_rate > 0 else 0.0
    time_start = max(0.0, float(args.time_start))
    time_end = min(duration, time_start + max(1e-6, float(args.time_span)))

    settings = AnalysisSettings()
    settings = AnalysisSettings(
        spectrogram=type(settings.spectrogram)(
            **{
                **settings.spectrogram.to_dict(),
                "freq_min": float(args.freq_min),
                "freq_max": float(args.freq_max),
                "preview_freq_max": float(args.freq_max),
            }
        ),
        snap=type(settings.snap)(
            **{
                **settings.snap.to_dict(),
                "freq_min": float(args.freq_min),
                "freq_max": float(args.freq_max),
            }
        ),
    )

    t0_init = time.perf_counter()
    renderer = SpectrogramRenderer(
        audio=audio,
        sample_rate=sample_rate,
        global_blend_octaves=settings.spectrogram.multires_blend_octaves,
        global_window_size_scale=settings.spectrogram.multires_window_size_scale,
    )
    renderer_init_ms = (time.perf_counter() - t0_init) * 1000.0

    labels = {
        "multires_stft": "Multi-Res STFT",
        "reassigned_stft": "librosa Reassigned STFT",
        "cwt": "ssqueezepy CWT",
        "fsst": "ssqueezepy FSST",
        "wsst": "ssqueezepy WSST",
    }
    method_order = ["multires_stft", "reassigned_stft", "cwt", "fsst", "wsst"]
    images: dict[str, np.ndarray] = {}
    timings: dict[str, list[float]] = {}
    extras: dict[str, Any] = {}

    try:
        time_span = max(1e-6, time_end - time_start)
        max_start = max(0.0, duration - time_span)

        def shifted_window(iteration: int) -> tuple[float, float]:
            shifted_start = min(max_start, time_start + (iteration * 0.003))
            return shifted_start, shifted_start + time_span

        (multires_result, multires_times) = _run_timed(
            lambda i: _render_multires_stft(
                renderer,
                settings,
                time_start=shifted_window(i)[0],
                time_end=shifted_window(i)[1],
                freq_min=float(args.freq_min),
                freq_max=float(args.freq_max),
                width=int(args.width),
                height=int(args.height),
            ),
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )
        images["multires_stft"], quality = multires_result
        timings["multires_stft"] = multires_times
        extras["multires_quality"] = quality

        reassigned_result, reassigned_times = _run_timed(
            lambda i: _render_reassigned_stft(
                audio,
                sample_rate,
                renderer,
                settings,
                time_start=shifted_window(i)[0],
                time_end=shifted_window(i)[1],
                freq_min=float(args.freq_min),
                freq_max=float(args.freq_max),
                width=int(args.width),
                height=int(args.height),
                sigma_time=float(args.reassigned_sigma_time),
                sigma_freq=float(args.reassigned_sigma_freq),
            ),
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )
        images["reassigned_stft"], reassigned_meta = reassigned_result
        timings["reassigned_stft"] = reassigned_times
        extras["reassigned_meta"] = reassigned_meta

        cwt_result, cwt_times = _run_timed(
            lambda i: _render_ssqueezepy_cwt(
                audio,
                sample_rate,
                time_start=shifted_window(i)[0],
                time_end=shifted_window(i)[1],
                freq_min=float(args.freq_min),
                freq_max=float(args.freq_max),
                width=int(args.width),
                height=int(args.height),
                settings=settings,
                nv=int(args.wsst_nv),
                use_sst=False,
                order=0,
            ),
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )
        images["cwt"] = cwt_result
        timings["cwt"] = cwt_times

        fsst_result, fsst_times = _run_timed(
            lambda i: _render_fsst(
                audio,
                sample_rate,
                time_start=shifted_window(i)[0],
                time_end=shifted_window(i)[1],
                freq_min=float(args.freq_min),
                freq_max=float(args.freq_max),
                width=int(args.width),
                height=int(args.height),
                settings=settings,
                n_fft=int(args.fsst_n_fft),
                hop_len=int(args.fsst_hop),
            ),
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )
        images["fsst"] = fsst_result
        timings["fsst"] = fsst_times

        wsst_result, wsst_times = _run_timed(
            lambda i: _render_ssqueezepy_cwt(
                audio,
                sample_rate,
                time_start=shifted_window(i)[0],
                time_end=shifted_window(i)[1],
                freq_min=float(args.freq_min),
                freq_max=float(args.freq_max),
                width=int(args.width),
                height=int(args.height),
                settings=settings,
                nv=int(args.wsst_nv),
                use_sst=True,
                order=0,
            ),
            warmup=int(args.warmup),
            iterations=int(args.iterations),
        )
        images["wsst"] = wsst_result
        timings["wsst"] = wsst_times
    finally:
        renderer.close()

    stats = {k: _stats(v) for k, v in timings.items()}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"spectrogram-methods-{stamp}.png"
    json_path = args.out_dir / f"spectrogram-methods-{stamp}.json"
    latest_png = args.out_dir / "spectrogram-methods-latest.png"
    latest_json = args.out_dir / "spectrogram-methods-latest.json"

    _plot(
        images=images,
        stats=stats,
        labels=labels,
        order=method_order,
        time_start=time_start,
        time_end=time_end,
        freq_min=float(args.freq_min),
        freq_max=float(args.freq_max),
        out_path=png_path,
    )
    latest_png.write_bytes(png_path.read_bytes())

    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "audio_path": str(args.audio),
        "sample_rate": sample_rate,
        "audio_duration_sec": duration,
        "window": {
            "time_start_sec": time_start,
            "time_end_sec": time_end,
            "freq_min_hz": float(args.freq_min),
            "freq_max_hz": float(args.freq_max),
            "width_px": int(args.width),
            "height_px": int(args.height),
        },
        "config": {
            "iterations": int(args.iterations),
            "warmup": int(args.warmup),
            "reassigned_sigma_time": float(args.reassigned_sigma_time),
            "reassigned_sigma_freq": float(args.reassigned_sigma_freq),
            "fsst_n_fft": int(args.fsst_n_fft),
            "fsst_hop": int(args.fsst_hop),
            "wsst_nv": int(args.wsst_nv),
            "renderer_init_ms": renderer_init_ms,
        },
        "timings_ms": timings,
        "stats_ms": stats,
        "extras": extras,
        "image_path": str(png_path),
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json.write_text(json_text, encoding="utf-8")
    print(png_path)
    print(json_path)


if __name__ == "__main__":
    main()
