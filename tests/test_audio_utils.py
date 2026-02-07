import numpy as np

from soma.audio_utils import time_stretch_pitch_preserving


def _dominant_freq(signal: np.ndarray, sample_rate: int) -> float:
    window = np.hanning(signal.size)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(np.abs(spectrum)))])


def test_time_stretch_preserves_pitch_and_changes_length() -> None:
    sample_rate = 48000
    duration_sec = 1.0
    samples = np.arange(int(sample_rate * duration_sec), dtype=np.float64)
    source = np.sin(2.0 * np.pi * 440.0 * samples / sample_rate).astype(np.float32)

    slow = time_stretch_pitch_preserving(source, speed_ratio=0.5, sample_rate=sample_rate)
    fast = time_stretch_pitch_preserving(source, speed_ratio=2.0, sample_rate=sample_rate)

    assert abs(slow.size - int(source.size / 0.5)) <= 4
    assert abs(fast.size - int(source.size / 2.0)) <= 4

    slow_center = slow[slow.size // 4 : slow.size // 4 + 16384]
    fast_center = fast[max(0, fast.size // 4) : max(0, fast.size // 4) + min(16384, fast.size // 2)]
    slow_freq = _dominant_freq(slow_center, sample_rate)
    fast_freq = _dominant_freq(fast_center, sample_rate)
    assert abs(slow_freq - 440.0) < 5.0
    assert abs(fast_freq - 440.0) < 5.0
