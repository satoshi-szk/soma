from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from soma.models import Partial

_PROBE_CROSSFADE_SEC = 0.02
_PROBE_SMOOTHING_SEC = 0.005


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
        self._probe_voice_ids: list[str] = []
        self._probe_runtime_freqs = np.array([], dtype=np.float64)
        self._probe_runtime_amps = np.array([], dtype=np.float64)
        self._probe_prev_freqs = np.array([], dtype=np.float64)
        self._probe_prev_amps = np.array([], dtype=np.float64)
        self._probe_prev_phases = np.array([], dtype=np.float64)
        self._probe_prev_runtime_freqs = np.array([], dtype=np.float64)
        self._probe_prev_runtime_amps = np.array([], dtype=np.float64)
        self._probe_crossfade_total = 0
        self._probe_crossfade_remaining = 0
        self._probe_stop_after_fade = False
        self._probe_gain = 1.0
        self._probe_prev_gain = 1.0
        self._master_gain = 1.0

    def load(self, buffer: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            self._buffer = buffer.astype(np.float32)
            self._sample_rate = sample_rate
            self._position = 0
            self._mode = None

    def update_buffer(self, buffer: np.ndarray, start_position_sec: float | None = None) -> None:
        # 再生中のバッファ差し替え時に短いクロスフェードを入れて、クリックノイズを抑える。
        with self._lock:
            new_buffer = buffer.astype(np.float32)
            if start_position_sec is not None:
                next_position = self._seconds_to_sample_for_size(start_position_sec, new_buffer.size)
            else:
                next_position = min(self._position, new_buffer.size)

            if self._playing and self._mode == "buffer" and self._buffer is not None:
                old_buffer = self._buffer
                old_position = min(self._position, old_buffer.size)
                fade_samples = max(1, int(self._sample_rate * 0.015))
                max_old = max(0, old_buffer.size - old_position)
                max_new = max(0, new_buffer.size - next_position)
                fade_samples = min(fade_samples, max_old, max_new)
                if fade_samples > 0:
                    old_head = old_buffer[old_position : old_position + fade_samples].astype(np.float64)
                    new_head = new_buffer[next_position : next_position + fade_samples].astype(np.float64)
                    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
                    blended = (old_head * (1.0 - ramp) + new_head * ramp).astype(np.float32)
                    new_buffer = new_buffer.copy()
                    new_buffer[next_position : next_position + fade_samples] = blended

            self._buffer = new_buffer
            self._position = next_position

    def set_master_volume(self, gain: float) -> None:
        with self._lock:
            self._master_gain = float(np.clip(gain, 0.0, 1.0))

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

    def play_probe(self, freqs: np.ndarray, amps: np.ndarray, voice_ids: list[str] | None = None) -> bool:
        with self._lock:
            if self._playing:
                return False
            self._probe_freqs = np.array([], dtype=np.float64)
            self._probe_amps = np.array([], dtype=np.float64)
            self._probe_phases = np.array([], dtype=np.float64)
            self._probe_voice_ids = []
            self._probe_runtime_freqs = np.array([], dtype=np.float64)
            self._probe_runtime_amps = np.array([], dtype=np.float64)
            self._probe_prev_freqs = np.array([], dtype=np.float64)
            self._probe_prev_amps = np.array([], dtype=np.float64)
            self._probe_prev_phases = np.array([], dtype=np.float64)
            self._probe_prev_runtime_freqs = np.array([], dtype=np.float64)
            self._probe_prev_runtime_amps = np.array([], dtype=np.float64)
            self._probe_prev_gain = 1.0
            self._begin_probe_transition(freqs, amps, voice_ids=voice_ids, stop_after_fade=False)
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

    def update_probe(self, freqs: np.ndarray, amps: np.ndarray, voice_ids: list[str] | None = None) -> bool:
        with self._lock:
            if not self._playing or self._mode != "probe":
                return False
            next_freqs = np.asarray(freqs, dtype=np.float64)
            next_amps = np.asarray(amps, dtype=np.float64)
            next_voice_ids = self._normalize_probe_voice_ids(voice_ids, next_freqs.size)
            # 同一ボイス構成の更新は遷移を張り直さず、目標パラメータのみ更新する。
            if next_voice_ids == self._probe_voice_ids:
                self._probe_freqs = next_freqs
                self._probe_amps = next_amps
                self._probe_gain = self._compute_probe_gain(self._probe_amps)
                self._probe_stop_after_fade = False
                return True
            self._begin_probe_transition(freqs, amps, voice_ids=voice_ids, stop_after_fade=False)
        return True

    def stop_probe(self) -> bool:
        with self._lock:
            if not self._playing or self._mode != "probe":
                return False
            self._begin_probe_transition(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                voice_ids=[],
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
        return self._seconds_to_sample_for_size(seconds, self._buffer.size)

    def _seconds_to_sample_for_size(self, seconds: float, size: int) -> int:
        if size <= 0:
            return 0
        clamped = float(np.clip(seconds, 0.0, size / float(self._sample_rate)))
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
                    new_block, self._probe_phases, self._probe_runtime_freqs, self._probe_runtime_amps = (
                        self._render_probe_bank_smoothed(
                            self._probe_freqs,
                            self._probe_amps,
                            self._probe_runtime_freqs,
                            self._probe_runtime_amps,
                            self._probe_phases,
                            fade_frames,
                            self._probe_gain,
                        )
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
                        self._probe_prev_runtime_freqs = np.array([], dtype=np.float64)
                        self._probe_prev_runtime_amps = np.array([], dtype=np.float64)
                        self._probe_prev_gain = 1.0

                if head < frames:
                    tail, self._probe_phases, self._probe_runtime_freqs, self._probe_runtime_amps = (
                        self._render_probe_bank_smoothed(
                            self._probe_freqs,
                            self._probe_amps,
                            self._probe_runtime_freqs,
                            self._probe_runtime_amps,
                            self._probe_phases,
                            frames - head,
                            self._probe_gain,
                        )
                    )
                    sample[head:] = tail

                outdata[:, 0] = sample.astype(np.float32)
                outdata[:, 0] *= self._master_gain
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
                    outdata[:, 0] *= self._master_gain
                    self._playing = False
                    self._mode = None
                    raise sd.CallbackStop
            outdata[:, 0] *= self._master_gain

    def _begin_probe_transition(
        self,
        freqs: np.ndarray,
        amps: np.ndarray,
        voice_ids: list[str] | None,
        stop_after_fade: bool,
    ) -> None:
        self._probe_prev_freqs = self._probe_freqs
        self._probe_prev_amps = self._probe_amps
        self._probe_prev_phases = self._probe_phases
        self._probe_prev_runtime_freqs = self._probe_runtime_freqs
        self._probe_prev_runtime_amps = self._probe_runtime_amps
        self._probe_prev_gain = self._probe_gain

        next_freqs = np.asarray(freqs, dtype=np.float64)
        next_amps = np.asarray(amps, dtype=np.float64)
        next_voice_ids = self._normalize_probe_voice_ids(voice_ids, next_freqs.size)
        next_phases = np.zeros(next_freqs.size, dtype=np.float64)
        next_runtime_freqs = next_freqs.copy()
        next_runtime_amps = next_amps.copy()
        prev_phase_by_voice = {
            voice_id: phase for voice_id, phase in zip(self._probe_voice_ids, self._probe_phases, strict=False)
        }
        prev_runtime_freq_by_voice = {
            voice_id: freq
            for voice_id, freq in zip(self._probe_voice_ids, self._probe_runtime_freqs, strict=False)
        }
        prev_runtime_amp_by_voice = {
            voice_id: amp for voice_id, amp in zip(self._probe_voice_ids, self._probe_runtime_amps, strict=False)
        }
        for index, voice_id in enumerate(next_voice_ids):
            phase = prev_phase_by_voice.get(voice_id)
            if phase is not None:
                next_phases[index] = phase
            prev_freq = prev_runtime_freq_by_voice.get(voice_id)
            if prev_freq is not None:
                next_runtime_freqs[index] = prev_freq
            prev_amp = prev_runtime_amp_by_voice.get(voice_id)
            if prev_amp is not None:
                next_runtime_amps[index] = prev_amp

        self._probe_freqs = next_freqs
        self._probe_amps = next_amps
        self._probe_voice_ids = next_voice_ids
        self._probe_phases = next_phases
        self._probe_runtime_freqs = next_runtime_freqs
        self._probe_runtime_amps = next_runtime_amps
        self._probe_gain = self._compute_probe_gain(self._probe_amps)

        self._probe_crossfade_total = max(1, int(self._sample_rate * _PROBE_CROSSFADE_SEC))
        self._probe_crossfade_remaining = self._probe_crossfade_total
        self._probe_stop_after_fade = stop_after_fade

    def _normalize_probe_voice_ids(self, voice_ids: list[str] | None, size: int) -> list[str]:
        if size <= 0:
            return []
        if voice_ids is None:
            return [f"voice-{index}" for index in range(size)]
        if len(voice_ids) >= size:
            return [str(voice_ids[index]) for index in range(size)]
        normalized = [str(voice_id) for voice_id in voice_ids]
        for index in range(len(normalized), size):
            normalized.append(f"voice-{index}")
        return normalized

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

    def _render_probe_bank_smoothed(
        self,
        target_freqs: np.ndarray,
        target_amps: np.ndarray,
        runtime_freqs: np.ndarray,
        runtime_amps: np.ndarray,
        phases: np.ndarray,
        frames: int,
        gain: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if target_freqs.size == 0 or target_amps.size == 0 or frames <= 0:
            return (
                np.zeros(frames, dtype=np.float64),
                phases,
                runtime_freqs,
                runtime_amps,
            )

        safe_runtime_freqs = np.asarray(runtime_freqs, dtype=np.float64)
        safe_runtime_amps = np.asarray(runtime_amps, dtype=np.float64)
        if safe_runtime_freqs.size != target_freqs.size:
            safe_runtime_freqs = target_freqs.copy()
        if safe_runtime_amps.size != target_amps.size:
            safe_runtime_amps = target_amps.copy()

        block_sec = frames / float(self._sample_rate)
        tau = max(_PROBE_SMOOTHING_SEC, 1e-6)
        alpha = 1.0 - np.exp(-block_sec / tau)
        next_runtime_freqs = safe_runtime_freqs + (target_freqs - safe_runtime_freqs) * alpha
        next_runtime_amps = safe_runtime_amps + (target_amps - safe_runtime_amps) * alpha

        ramp = (np.arange(frames, dtype=np.float64)[:, None] + 1.0) / float(frames)
        freq_ramp = safe_runtime_freqs[None, :] + ramp * (next_runtime_freqs - safe_runtime_freqs)[None, :]
        amp_ramp = safe_runtime_amps[None, :] + ramp * (next_runtime_amps - safe_runtime_amps)[None, :]
        phase_steps = (2.0 * np.pi * freq_ramp) / self._sample_rate
        phase_matrix = phases[None, :] + np.cumsum(phase_steps, axis=0)
        block = np.sum(np.sin(phase_matrix) * amp_ramp, axis=1)
        next_phases = np.mod(phase_matrix[-1, :], 2.0 * np.pi)
        return block * gain, next_phases, next_runtime_freqs, next_runtime_amps

    def _compute_probe_gain(self, amps: np.ndarray) -> float:
        amp_sum = float(np.sum(np.abs(amps)))
        return 1.0 if amp_sum <= 1.0 else 1.0 / amp_sum
