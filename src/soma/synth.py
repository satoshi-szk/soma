from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from soma.models import Partial

_PROBE_CROSSFADE_SEC = 0.02


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
        self._mode: str | None = None
        self._position = 0
        self._loop = False
        self._playing = False
        self._sample_rate = 44100
        self._probe_freqs = np.array([], dtype=np.float64)
        self._probe_amps = np.array([], dtype=np.float64)
        self._probe_phases = np.array([], dtype=np.float64)
        self._probe_prev_freqs = np.array([], dtype=np.float64)
        self._probe_prev_amps = np.array([], dtype=np.float64)
        self._probe_prev_phases = np.array([], dtype=np.float64)
        self._probe_crossfade_total = 0
        self._probe_crossfade_remaining = 0
        self._probe_stop_after_fade = False
        self._probe_gain = 1.0
        self._probe_prev_gain = 1.0

    def load(self, buffer: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            self._buffer = buffer.astype(np.float32)
            self._sample_rate = sample_rate
            self._position = 0
            self._mode = None

    def play(self, loop: bool = False, start_position_sec: float = 0.0) -> None:
        with self._lock:
            if self._buffer is None or self._playing:
                return
            self._loop = loop
            self._position = self._seconds_to_sample(start_position_sec)
            self._mode = "buffer"
            self._playing = True
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

    def play_probe(self, freqs: np.ndarray, amps: np.ndarray) -> bool:
        with self._lock:
            if self._playing:
                return False
            self._probe_freqs = np.array([], dtype=np.float64)
            self._probe_amps = np.array([], dtype=np.float64)
            self._probe_phases = np.array([], dtype=np.float64)
            self._probe_prev_freqs = np.array([], dtype=np.float64)
            self._probe_prev_amps = np.array([], dtype=np.float64)
            self._probe_prev_phases = np.array([], dtype=np.float64)
            self._probe_prev_gain = 1.0
            self._begin_probe_transition(freqs, amps, stop_after_fade=False)
            self._mode = "probe"
            self._playing = True
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        return True

    def update_probe(self, freqs: np.ndarray, amps: np.ndarray) -> bool:
        with self._lock:
            if not self._playing or self._mode != "probe":
                return False
            self._begin_probe_transition(freqs, amps, stop_after_fade=False)
        return True

    def stop_probe(self) -> bool:
        with self._lock:
            if not self._playing or self._mode != "probe":
                return False
            self._begin_probe_transition(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                stop_after_fade=True,
            )
        return True

    def stop(self, reset_position_sec: float | None = 0.0) -> None:
        with self._lock:
            if reset_position_sec is not None:
                self._position = self._seconds_to_sample(reset_position_sec)
            self._mode = None
        self._stop_stream()

    def pause(self) -> None:
        with self._lock:
            self._mode = None
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
            if self._mode == "probe":
                sample = np.zeros(frames, dtype=np.float64)
                head = 0
                if self._probe_crossfade_remaining > 0 and self._probe_crossfade_total > 0:
                    fade_frames = min(frames, self._probe_crossfade_remaining)
                    old_block, self._probe_prev_phases = self._render_probe_bank(
                        self._probe_prev_freqs,
                        self._probe_prev_amps,
                        self._probe_prev_phases,
                        fade_frames,
                        self._probe_prev_gain,
                    )
                    new_block, self._probe_phases = self._render_probe_bank(
                        self._probe_freqs,
                        self._probe_amps,
                        self._probe_phases,
                        fade_frames,
                        self._probe_gain,
                    )
                    ramp_start = self._probe_crossfade_total - self._probe_crossfade_remaining
                    fade = (np.arange(fade_frames, dtype=np.float64) + ramp_start + 1.0) / self._probe_crossfade_total
                    sample[:fade_frames] = old_block * (1.0 - fade) + new_block * fade
                    self._probe_crossfade_remaining -= fade_frames
                    head = fade_frames
                    if self._probe_crossfade_remaining == 0:
                        self._probe_prev_freqs = np.array([], dtype=np.float64)
                        self._probe_prev_amps = np.array([], dtype=np.float64)
                        self._probe_prev_phases = np.array([], dtype=np.float64)
                        self._probe_prev_gain = 1.0

                if head < frames:
                    tail, self._probe_phases = self._render_probe_bank(
                        self._probe_freqs,
                        self._probe_amps,
                        self._probe_phases,
                        frames - head,
                        self._probe_gain,
                    )
                    sample[head:] = tail

                outdata[:, 0] = sample.astype(np.float32)
                if (
                    self._probe_stop_after_fade
                    and self._probe_crossfade_remaining == 0
                    and self._probe_freqs.size == 0
                ):
                    self._playing = False
                    self._mode = None
                    self._probe_stop_after_fade = False
                    raise sd.CallbackStop
                return
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
                    tail_samples = remaining % buffer.size
                    if tail_samples > 0:
                        outdata[offset : offset + tail_samples, 0] = buffer[:tail_samples]
                    self._position = tail_samples
                else:
                    outdata[available:, 0] = 0
                    self._position = buffer.size
                    self._playing = False
                    self._mode = None
                    raise sd.CallbackStop

    def _begin_probe_transition(self, freqs: np.ndarray, amps: np.ndarray, stop_after_fade: bool) -> None:
        self._probe_prev_freqs = self._probe_freqs
        self._probe_prev_amps = self._probe_amps
        self._probe_prev_phases = self._probe_phases
        self._probe_prev_gain = self._probe_gain

        self._probe_freqs = np.asarray(freqs, dtype=np.float64)
        self._probe_amps = np.asarray(amps, dtype=np.float64)
        self._probe_phases = np.zeros(self._probe_freqs.size, dtype=np.float64)
        self._probe_gain = self._compute_probe_gain(self._probe_amps)

        self._probe_crossfade_total = max(1, int(self._sample_rate * _PROBE_CROSSFADE_SEC))
        self._probe_crossfade_remaining = self._probe_crossfade_total
        self._probe_stop_after_fade = stop_after_fade

    def _render_probe_bank(
        self,
        freqs: np.ndarray,
        amps: np.ndarray,
        phases: np.ndarray,
        frames: int,
        gain: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if freqs.size == 0 or amps.size == 0 or frames <= 0:
            return np.zeros(frames, dtype=np.float64), phases
        phase_step = (2.0 * np.pi * freqs) / self._sample_rate
        frame_offsets = np.arange(frames, dtype=np.float64)[:, None]
        phase_matrix = phases[None, :] + frame_offsets * phase_step[None, :]
        block = np.sum(np.sin(phase_matrix) * amps[None, :], axis=1)
        next_phases = np.mod(phases + phase_step * frames, 2.0 * np.pi)
        return block * gain, next_phases

    def _compute_probe_gain(self, amps: np.ndarray) -> float:
        amp_sum = float(np.sum(np.abs(amps)))
        return 1.0 if amp_sum <= 1.0 else 1.0 / amp_sum
