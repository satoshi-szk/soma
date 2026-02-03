from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from soma.models import Partial


@dataclass
class RenderedPartial:
    start: int
    data: np.ndarray


class Synthesizer:
    def __init__(self, sample_rate: int, duration_sec: float) -> None:
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.total_samples = max(1, int(duration_sec * sample_rate))
        self.master = np.zeros(self.total_samples, dtype=np.float64)
        self._render_cache: dict[str, RenderedPartial] = {}
        self._lock = threading.Lock()

    def reset(self, sample_rate: int, duration_sec: float) -> None:
        with self._lock:
            self.sample_rate = sample_rate
            self.duration_sec = duration_sec
            self.total_samples = max(1, int(duration_sec * sample_rate))
            self.master = np.zeros(self.total_samples, dtype=np.float64)
            self._render_cache.clear()

    def apply_partial(self, partial: Partial) -> None:
        if partial.is_muted:
            self.remove_partial(partial.id)
            return
        self.remove_partial(partial.id)
        rendered = render_partial(partial, self.sample_rate, self.total_samples)
        if rendered is None:
            return
        with self._lock:
            self.master[rendered.start : rendered.start + rendered.data.size] += rendered.data
            self._render_cache[partial.id] = rendered

    def remove_partial(self, partial_id: str) -> None:
        with self._lock:
            rendered = self._render_cache.pop(partial_id, None)
            if rendered is None:
                return
            self.master[rendered.start : rendered.start + rendered.data.size] -= rendered.data

    def rebuild(self, partials: Iterable[Partial]) -> None:
        with self._lock:
            self.master.fill(0.0)
            self._render_cache.clear()
        for partial in partials:
            self.apply_partial(partial)

    def get_mix_buffer(self) -> np.ndarray:
        with self._lock:
            return self.master.copy()


def render_partial(partial: Partial, sample_rate: int, total_samples: int) -> RenderedPartial | None:
    points = partial.sorted_points()
    if len(points) < 2:
        return None

    times = np.array([p.time for p in points], dtype=np.float64)
    freqs = np.array([p.freq for p in points], dtype=np.float64)
    amps = np.array([p.amp for p in points], dtype=np.float64)

    start_time = max(0.0, float(times[0]))
    end_time = min(float(times[-1]), total_samples / sample_rate)
    if end_time <= start_time:
        return None

    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)
    if end_idx <= start_idx:
        return None

    sample_times = (np.arange(end_idx - start_idx) / sample_rate) + start_time
    freq_interp = np.interp(sample_times, times, freqs)
    amp_interp = np.interp(sample_times, times, amps)

    phase = 2.0 * np.pi * np.cumsum(freq_interp) / sample_rate
    wave = np.sin(phase) * amp_interp

    fade_samples = max(1, int(sample_rate * 0.005))
    fade_samples = min(fade_samples, wave.size // 2)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out

    return RenderedPartial(start=start_idx, data=wave.astype(np.float64))


class AudioPlayer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._buffer: np.ndarray | None = None
        self._position = 0
        self._loop = False
        self._playing = False
        self._sample_rate = 44100

    def load(self, buffer: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            self._buffer = buffer.astype(np.float32)
            self._sample_rate = sample_rate
            self._position = 0

    def play(self, loop: bool = False, start_position_sec: float = 0.0) -> None:
        with self._lock:
            if self._buffer is None or self._playing:
                return
            self._loop = loop
            self._position = self._seconds_to_sample(start_position_sec)
            self._playing = True
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

    def stop(self, reset_position_sec: float | None = 0.0) -> None:
        with self._lock:
            if reset_position_sec is not None:
                self._position = self._seconds_to_sample(reset_position_sec)
        self._stop_stream()

    def pause(self) -> None:
        self._stop_stream()

    def _stop_stream(self) -> None:
        with self._lock:
            if self._stream is None:
                return
            stream = self._stream
            self._stream = None
            self._playing = False
        try:
            stream.stop()
        finally:
            stream.close()

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def position_sec(self) -> float:
        with self._lock:
            return self._position / float(self._sample_rate)

    def _seconds_to_sample(self, seconds: float) -> int:
        if self._buffer is None:
            return 0
        clamped = float(np.clip(seconds, 0.0, self._buffer.size / float(self._sample_rate)))
        return int(clamped * self._sample_rate)

    def _callback(self, outdata: np.ndarray, frames: int, time: object, status: object) -> None:
        del time, status
        with self._lock:
            if self._buffer is None:
                outdata[:] = 0
                return
            buffer = self._buffer
            start = self._position
            end = start + frames
            if end <= buffer.size:
                outdata[:, 0] = buffer[start:end]
                self._position = end
            else:
                available = max(0, buffer.size - start)
                if available > 0:
                    outdata[:available, 0] = buffer[start:]
                if self._loop and buffer.size > 0:
                    remaining = frames - available
                    repeats = remaining // buffer.size
                    offset = available
                    for _ in range(repeats):
                        outdata[offset : offset + buffer.size, 0] = buffer
                        offset += buffer.size
                    tail = remaining % buffer.size
                    if tail > 0:
                        outdata[offset : offset + tail, 0] = buffer[:tail]
                    self._position = tail
                else:
                    outdata[available:, 0] = 0
                    self._position = buffer.size
                    self._playing = False
                    raise sd.CallbackStop
