from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, stft


@dataclass(frozen=True)
class AudioInfo:
    path: str
    name: str
    sample_rate: int
    duration_sec: float
    channels: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectrogramPreview:
    width: int
    height: int
    data: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_audio(path: Path, max_duration_sec: float = 30.0) -> tuple[AudioInfo, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    if raw.ndim == 1:
        channels = 1
        audio = raw
    else:
        channels = raw.shape[1]
        audio = raw.mean(axis=1)

    audio = _to_float32(audio)
    total_samples = audio.shape[0]
    max_samples = int(sample_rate * max_duration_sec)
    truncated = total_samples > max_samples
    if truncated:
        audio = audio[:max_samples]

    duration_sec = total_samples / float(sample_rate)
    info = AudioInfo(
        path=str(path),
        name=path.name,
        sample_rate=int(sample_rate),
        duration_sec=duration_sec,
        channels=channels,
        truncated=truncated,
    )
    return info, audio


def make_spectrogram_preview(
    audio: np.ndarray,
    sample_rate: int,
    width: int = 512,
    height: int = 256,
) -> SpectrogramPreview:
    if audio.size == 0:
        return SpectrogramPreview(width=width, height=height, data=[0] * (width * height))

    nperseg = min(1024, audio.shape[0])
    noverlap = int(nperseg * 0.75)
    _, _, zxx = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hann")
    magnitude = np.abs(zxx).astype(np.float32)

    magnitude = np.log10(magnitude + 1e-6)
    magnitude -= float(magnitude.min())
    magnitude /= float(magnitude.max() + 1e-8)

    resized = resample(magnitude, height, axis=0)
    resized = resample(resized, width, axis=1)
    normalized = np.clip(resized, 0.0, 1.0)

    data = (normalized * 255).astype(np.uint8).flatten().tolist()
    return SpectrogramPreview(width=width, height=height, data=data)


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        return (audio.astype(np.float32) / float(info.max)).clip(-1.0, 1.0)

    return audio.astype(np.float32)
