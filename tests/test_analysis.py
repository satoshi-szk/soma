import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.io import wavfile

pytest.importorskip("pywt")

from soma.analysis import (  # noqa: E402
    _read_audio_audioread,
    estimate_cwt_amp_reference,
    load_audio,
    make_spectrogram,
    make_spectrogram_stft,
    snap_trace,
)
from soma.models import AnalysisSettings  # noqa: E402


def _write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    wavfile.write(path, sample_rate, data)


def test_load_audio_truncates(tmp_path: Path) -> None:
    path = tmp_path / "test.wav"
    sample_rate = 44100
    duration_sec = 2.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    data = (0.5 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
    _write_wav(path, sample_rate, data)

    info, audio = load_audio(path, max_duration_sec=0.5)

    assert info.truncated is True
    assert info.sample_rate == sample_rate
    assert audio.shape[0] <= int(sample_rate * 0.5)


def test_load_audio_normalizes_stereo_int32(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    sample_rate = 44100
    peak = np.iinfo(np.int32).max
    data = np.full((10, 2), peak, dtype=np.int32)
    _write_wav(path, sample_rate, data)

    _info, audio = load_audio(path, max_duration_sec=None)

    assert audio.ndim == 1
    assert audio.max() <= 1.0
    assert audio.min() >= -1.0


def test_load_audio_reads_aiff(tmp_path: Path) -> None:
    path = tmp_path / "test.aiff"
    sample_rate = 48000
    data = np.random.uniform(-0.5, 0.5, size=(2000, 2)).astype(np.float32)
    sf.write(path, data, sample_rate, format="AIFF")

    info, audio = load_audio(path, max_duration_sec=None)

    assert info.sample_rate == sample_rate
    assert info.channels == 2
    assert audio.ndim == 1
    assert audio.shape[0] == data.shape[0]


def test_load_audio_accepts_display_name(tmp_path: Path) -> None:
    path = tmp_path / "temp-name.wav"
    sample_rate = 44100
    data = np.zeros(sample_rate // 10, dtype=np.float32)
    _write_wav(path, sample_rate, data)

    info, _audio = load_audio(path, max_duration_sec=None, display_name="original.wav")

    assert info.path == str(path)
    assert info.name == "original.wav"


def test_audioread_frame_alignment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import audioread

    sample_rate = 48000
    channels = 2
    left = np.array([0, 1000, -1000, 2000, -2000, 3000], dtype=np.int16)
    right = np.array([100, -100, 200, -200, 300, -300], dtype=np.int16)
    interleaved = np.empty(left.size * 2, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right
    raw = interleaved.tobytes()

    buffers = [raw[:3], raw[3:11], raw[11:19], raw[19:]]

    class FakeHandle:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks
            self.samplerate = sample_rate
            self.channels = channels
            self.duration = 0.0

        def __iter__(self):
            return iter(self._chunks)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_open(_path: str):
        return FakeHandle(buffers)

    monkeypatch.setattr(audioread, "audio_open", fake_open)
    fake_path = tmp_path / "fake.mp3"

    _rate, _channels, _duration, _truncated, audio = _read_audio_audioread(fake_path, None)

    expected = ((left.astype(np.float32) + right.astype(np.float32)) / 2.0) / 32768.0
    assert audio.shape[0] == expected.shape[0]
    assert np.allclose(audio, expected, atol=1e-6)


def test_make_spectrogram() -> None:
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
    settings = AnalysisSettings(time_resolution_ms=0.0)

    preview, amp_reference = make_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        settings=settings,
        time_start=0.0,
        time_end=duration,
        freq_min=settings.freq_min,
        freq_max=min(settings.preview_freq_max, settings.freq_max),
        width=768,
        height=320,
    )

    assert preview.width == 768
    assert preview.height == 320
    assert len(preview.data) == preview.width * preview.height
    assert preview.duration_sec > 0.0
    assert preview.time_start == 0.0
    assert preview.time_end == duration
    assert amp_reference > 0.0


def test_make_spectrogram_stft() -> None:
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, sample_rate, endpoint=False)
    audio = (0.2 * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
    settings = AnalysisSettings(time_resolution_ms=0.0)

    preview, amp_reference = make_spectrogram_stft(
        audio=audio,
        sample_rate=sample_rate,
        settings=settings,
        time_start=0.0,
        time_end=duration,
        freq_min=settings.freq_min,
        freq_max=min(settings.preview_freq_max, settings.freq_max),
        width=256,
        height=128,
    )

    assert preview.width == 256
    assert preview.height == 128
    assert len(preview.data) == preview.width * preview.height
    assert preview.duration_sec > 0.0
    assert amp_reference > 0.0


def test_make_spectrogram_preserves_freq_max_for_longer_window() -> None:
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.2 * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)
    settings = AnalysisSettings(time_resolution_ms=0.0)

    preview, _amp_reference = make_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        settings=settings,
        time_start=0.0,
        time_end=duration,
        freq_min=settings.freq_min,
        freq_max=min(settings.preview_freq_max, settings.freq_max),
        width=120,
        height=64,
    )

    assert preview.freq_max > 5000.0


def test_estimate_cwt_amp_reference_uses_stft_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    from soma import analysis

    def fake_peak_location(
        _audio: np.ndarray,
        _sample_rate: int,
        _freq_min: float,
        _freq_max: float,
        width: int,
    ) -> tuple[int, float]:
        assert width == 64
        return 450, 440.0

    def fake_build_frequencies(_: AnalysisSettings, max_freq: float | None = None) -> np.ndarray:
        return np.array([100.0], dtype=np.float32)

    captured: dict[str, np.ndarray] = {}

    def fake_cwt_magnitude(
        audio: np.ndarray,
        _sample_rate: float,
        _frequencies: np.ndarray,
        wavelet_bandwidth: float = 8.0,
        wavelet_center_freq: float = 1.0,
    ) -> np.ndarray:
        captured["window"] = audio
        return np.array([[np.max(audio)]], dtype=np.float32)

    monkeypatch.setattr(analysis, "_stft_peak_location", fake_peak_location)
    monkeypatch.setattr(analysis, "_build_frequencies", fake_build_frequencies)
    monkeypatch.setattr(analysis, "_cwt_magnitude", fake_cwt_magnitude)

    audio = np.zeros(1000, dtype=np.float32)
    audio[400:500] = 1.0
    settings = AnalysisSettings()

    amp_reference = estimate_cwt_amp_reference(
        audio,
        sample_rate=1000,
        settings=settings,
        stft_width=64,
        window_ms=100.0,
    )

    assert amp_reference == pytest.approx(1.0)
    assert captured["window"].shape[0] == 100
    assert np.allclose(captured["window"], 1.0)


def test_snap_trace_returns_points() -> None:
    sample_rate = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.8 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
    settings = AnalysisSettings(time_resolution_ms=0.0)
    trace = [(0.1, 450.0), (0.2, 445.0), (0.3, 430.0)]

    points = snap_trace(audio, sample_rate, settings, trace)

    assert len(points) == len(trace)
    for point in points:
        assert 400.0 < point.freq < 500.0
        assert 0.0 <= point.amp <= 1.0


def test_snap_trace_uses_amp_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    from soma import analysis

    def fake_build_frequencies(_: AnalysisSettings, max_freq: float | None = None) -> np.ndarray:
        return np.array([100.0, 200.0], dtype=np.float32)

    def fake_cwt_magnitude(
        _audio: np.ndarray,
        _sample_rate: float,
        _frequencies: np.ndarray,
        wavelet_bandwidth: float = 8.0,
        wavelet_center_freq: float = 1.0,
    ) -> np.ndarray:
        return np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(analysis, "_build_frequencies", fake_build_frequencies)
    monkeypatch.setattr(analysis, "_cwt_magnitude", fake_cwt_magnitude)

    audio = np.ones(1000, dtype=np.float32)
    settings = AnalysisSettings()
    trace = [(0.1, 150.0)]

    points = analysis.snap_trace(audio, 1000, settings, trace, amp_reference=4.0)

    assert len(points) == 1
    assert points[0].amp == pytest.approx(0.5)


def test_snap_trace_decimates_points() -> None:
    pytest.skip("snap_trace no longer decimates points for quality reasons")


def test_snap_trace_resamples_to_time_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    from soma import analysis

    def fake_build_frequencies(_: AnalysisSettings, max_freq: float | None = None) -> np.ndarray:
        return np.array([100.0, 200.0], dtype=np.float32)

    def fake_cwt_magnitude(
        audio: np.ndarray,
        _sample_rate: float,
        _frequencies: np.ndarray,
        wavelet_bandwidth: float = 8.0,
        wavelet_center_freq: float = 1.0,
    ) -> np.ndarray:
        return np.vstack([np.ones(audio.size, dtype=np.float32), np.ones(audio.size, dtype=np.float32) * 2.0])

    monkeypatch.setattr(analysis, "_build_frequencies", fake_build_frequencies)
    monkeypatch.setattr(analysis, "_cwt_magnitude", fake_cwt_magnitude)

    audio = np.ones(1000, dtype=np.float32)
    settings = AnalysisSettings(time_resolution_ms=50.0)
    trace = [(0.0, 150.0), (0.2, 150.0)]

    points = analysis.snap_trace(audio, 1000, settings, trace)

    assert len(points) == 5


def test_cwt_ridge_is_not_too_broad_for_sine() -> None:
    """Guardrail for visualization/peak-finding defaults.

    A pure sine should not show a strong ridge an octave above the fundamental.
    """
    from soma import analysis

    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.8 * np.sin(2 * math.pi * 1000.0 * t)).astype(np.float32)
    settings = AnalysisSettings(freq_min=200.0, freq_max=8000.0, bins_per_octave=48)

    frequencies = analysis._build_frequencies(settings, max_freq=settings.freq_max)
    magnitude = analysis._cwt_magnitude(
        audio,
        sample_rate,
        frequencies,
        wavelet_bandwidth=settings.wavelet_bandwidth,
        wavelet_center_freq=settings.wavelet_center_freq,
    )
    spectrum = magnitude[:, magnitude.shape[1] // 2]
    peak = float(np.max(spectrum))
    assert peak > 0

    # Compare level at ~2kHz vs peak.
    idx_2k = int(np.argmin(np.abs(frequencies - 2000.0)))
    rel_db = 20.0 * math.log10(float(spectrum[idx_2k]) / peak + 1e-12)
    assert rel_db < -45.0
