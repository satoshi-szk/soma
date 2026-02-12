from __future__ import annotations

import threading

import numpy as np

from soma.audio_utils import peak_normalize_buffer, time_stretch_pitch_preserving
from soma.exporter import MidiExportSettings, build_midi_for_playback
from soma.models import Partial, PlaybackSettings
from soma.services.document_utils import partial_sample_at_time
from soma.session import ProjectSession

_MIDI_CC_UPDATE_RATE_OPTIONS_HZ = (50, 100, 200, 400, 800)


class PlaybackService:
    def __init__(self, session: ProjectSession) -> None:
        self._session = session

    def set_master_volume(self, master_volume: float) -> float:
        clamped = float(np.clip(master_volume, 0.0, 1.0))
        self._session._master_volume = clamped
        self._session.player.set_master_volume(clamped)
        return clamped

    def master_volume(self) -> float:
        return self._session._master_volume

    def playback_settings(self) -> PlaybackSettings:
        return PlaybackSettings(
            master_volume=self._session._master_volume,
            output_mode=self._session._playback_output_mode,
            mix_ratio=self._session._last_mix_ratio,
            speed_ratio=self._session._last_speed_ratio,
            time_stretch_mode=self._session._last_time_stretch_mode,
            midi_mode=self._session._midi_mode,
            midi_output_name=self._session._midi_output_name,
            midi_pitch_bend_range=self._session._midi_pitch_bend_range,
            midi_amplitude_mapping=self._session._midi_amplitude_mapping,
            midi_amplitude_curve=self._session._midi_amplitude_curve,
            midi_cc_update_rate_hz=self._session._midi_cc_update_rate_hz,
            midi_bpm=self._session._midi_bpm,
        )

    def midi_outputs(self) -> list[str]:
        return self._session.midi_player.list_outputs()

    def last_list_outputs_error(self) -> str | None:
        return self._session.midi_player.last_list_outputs_error()

    def update_playback_settings(self, settings: PlaybackSettings) -> PlaybackSettings:
        if settings.output_mode != self._session._playback_output_mode:
            self.stop(return_position_sec=None)
        self._session._playback_output_mode = settings.output_mode
        self._session._last_mix_ratio = float(np.clip(settings.mix_ratio, 0.0, 1.0))
        self._session._last_speed_ratio = float(np.clip(settings.speed_ratio, 0.125, 8.0))
        self._session._last_time_stretch_mode = settings.time_stretch_mode
        self._session._midi_mode = settings.midi_mode
        self._session._midi_output_name = settings.midi_output_name
        self._session._midi_pitch_bend_range = max(1, int(settings.midi_pitch_bend_range))
        self._session._midi_amplitude_mapping = settings.midi_amplitude_mapping
        self._session._midi_amplitude_curve = settings.midi_amplitude_curve
        requested_cc_rate = int(settings.midi_cc_update_rate_hz)
        self._session._midi_cc_update_rate_hz = min(
            _MIDI_CC_UPDATE_RATE_OPTIONS_HZ,
            key=lambda rate: abs(rate - requested_cc_rate),
        )
        self._session._midi_bpm = max(1.0, float(settings.midi_bpm))
        return self.playback_settings()

    def play(
        self,
        mix_ratio: float,
        loop: bool,
        start_position_sec: float | None = None,
        speed_ratio: float = 1.0,
        time_stretch_mode: str = "librosa",
    ) -> None:
        if self._session.audio_info is None:
            return
        if self.is_resynthesizing():
            return
        self._session._last_mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        clamped_speed = float(np.clip(speed_ratio, 0.125, 8.0))
        self._session._last_speed_ratio = clamped_speed
        self._session._last_time_stretch_mode = time_stretch_mode
        self._cancel_playback_prepare()
        if self._session.player.is_playing():
            self._session.player.stop(reset_position_sec=None)
        if self._session.midi_player.is_playing():
            self._session.midi_player.stop()
        self._session._playback_mode = None
        self._session._playback_speed_ratio = 1.0
        start_sec = float(start_position_sec or 0.0)
        self._session._pending_playback_position_sec = start_sec
        self._session._playback_speed_ratio = clamped_speed
        if self._session._playback_output_mode == "midi":
            settings = MidiExportSettings(
                pitch_bend_range=self._session._midi_pitch_bend_range,
                amplitude_mapping=self._session._midi_amplitude_mapping,
                amplitude_curve=self._session._midi_amplitude_curve,
                cc_update_rate_hz=self._session._midi_cc_update_rate_hz,
                bpm=self._session._midi_bpm,
            )
            midi = build_midi_for_playback(self._session.store.all(), self._session._midi_mode, settings)
            if not self._session._midi_output_name:
                raise ValueError("No MIDI output device selected.")
            self._session.midi_player.play_until(
                midi=midi,
                output_name=self._session._midi_output_name,
                start_position_sec=start_sec,
                speed_ratio=clamped_speed,
                end_position_sec=self._session.audio_info.duration_sec,
            )
            self._session._playback_mode = "normal"
            return
        self._session._playback_mode = "normal"
        if abs(clamped_speed - 1.0) <= 1e-4:
            mixed = self._mix_buffer(self._session._last_mix_ratio)
            self._session.player.load(peak_normalize_buffer(mixed), self._session.audio_info.sample_rate)
            self._session.player.play(loop=loop, start_position_sec=start_sec)
            return
        cache_key = self._playback_cache_lookup_key(self._session._last_mix_ratio, clamped_speed, time_stretch_mode)
        cached = self._lookup_playback_cache(cache_key)
        if cached is not None:
            self._session.player.load(cached, self._session.audio_info.sample_rate)
            self._session.player.play(loop=loop, start_position_sec=start_sec / clamped_speed)
            return
        with self._session._lock:
            self._session._playback_prepare_request_id += 1
            request_id = self._session._playback_prepare_request_id
            self._session._is_preparing_playback = True
        sample_rate = self._session.audio_info.sample_rate

        def _worker() -> None:
            try:
                prepared = self._build_speed_adjusted_buffer(
                    self._session._last_mix_ratio,
                    clamped_speed,
                    sample_rate,
                    time_stretch_mode,
                )
                with self._session._lock:
                    if request_id != self._session._playback_prepare_request_id:
                        return
                    self._session._is_preparing_playback = False
                self._store_playback_cache(cache_key, prepared)
                self._session.player.load(prepared, sample_rate)
                self._session.player.play(loop=loop, start_position_sec=start_sec / clamped_speed)
            except Exception:
                self._session._logger.exception("playback prepare failed")
                with self._session._lock:
                    if request_id != self._session._playback_prepare_request_id:
                        return
                    self._session._is_preparing_playback = False
                    self._session._playback_mode = None
                    self._session._playback_speed_ratio = 1.0

        thread = threading.Thread(target=_worker, name="soma-playback-prepare", daemon=True)
        thread.start()

    def start_harmonic_probe(self, time_sec: float) -> bool:
        if self._session.audio_info is None:
            return False
        self._cancel_playback_prepare()
        partial_ids, freqs, amps = self._harmonic_probe_state(time_sec)
        if self._session.player.is_playing():
            self._session.player.stop(reset_position_sec=None)
        if self._session.midi_player.is_playing():
            self._session.midi_player.stop()
        if self._session._playback_output_mode == "midi":
            if not self._session._midi_output_name:
                raise ValueError("No MIDI output device selected.")
            started = self._session.midi_player.start_probe(
                output_name=self._session._midi_output_name,
                partial_ids=partial_ids,
                freqs=freqs.tolist(),
                amps=amps.tolist(),
                pitch_bend_range=self._session._midi_pitch_bend_range,
                midi_mode=self._session._midi_mode,
                amplitude_mapping=self._session._midi_amplitude_mapping,
                amplitude_curve=self._session._midi_amplitude_curve,
            )
        else:
            started = self._session.player.play_probe(freqs, amps, voice_ids=partial_ids)
        if started:
            self._session._playback_mode = "probe"
            self._session._playback_speed_ratio = 1.0
        return started

    def update_harmonic_probe(self, time_sec: float) -> bool:
        if self._session._playback_mode != "probe":
            return False
        partial_ids, freqs, amps = self._harmonic_probe_state(time_sec)
        if self._session._playback_output_mode == "midi":
            return self._session.midi_player.update_probe(
                partial_ids=partial_ids,
                freqs=freqs.tolist(),
                amps=amps.tolist(),
                midi_mode=self._session._midi_mode,
                amplitude_mapping=self._session._midi_amplitude_mapping,
                amplitude_curve=self._session._midi_amplitude_curve,
            )
        return self._session.player.update_probe(freqs, amps, voice_ids=partial_ids)

    def stop_harmonic_probe(self) -> None:
        if self._session._playback_mode != "probe":
            return
        if self._session._playback_output_mode == "midi":
            self._session.midi_player.stop_probe()
        else:
            stopped = self._session.player.stop_probe()
            if not stopped:
                self._session.player.stop(reset_position_sec=None)
        self._session._playback_mode = None
        self._session._playback_speed_ratio = 1.0

    def pause(self) -> None:
        self._cancel_playback_prepare()
        self._session.player.pause()
        self._session.midi_player.stop()
        self._session._playback_mode = None
        self._session._playback_speed_ratio = 1.0

    def stop(self, return_position_sec: float | None = 0.0) -> None:
        self._cancel_playback_prepare()
        if self._session._playback_output_mode == "midi":
            self._session.midi_player.stop()
        else:
            reset_position = (
                None if return_position_sec is None else return_position_sec / self._session._playback_speed_ratio
            )
            self._session.player.stop(reset_position_sec=reset_position)
        self._session._playback_mode = None
        self._session._playback_speed_ratio = 1.0

    def playback_position(self) -> float:
        with self._session._lock:
            if self._session._is_preparing_playback:
                return self._session._pending_playback_position_sec
        if self._session._playback_output_mode == "midi":
            return self._session.midi_player.position_sec()
        return self._session.player.position_sec() * self._session._playback_speed_ratio

    def is_playing(self) -> bool:
        if self._session._playback_mode != "normal":
            return False
        if self._session._playback_output_mode == "midi":
            return self._session.midi_player.is_playing()
        return self._session.player.is_playing()

    def is_probe_playing(self) -> bool:
        if self._session._playback_mode != "probe":
            return False
        if self._session._playback_output_mode == "midi":
            return self._session.midi_player.is_playing()
        return self._session.player.is_playing()

    def is_preparing_playback(self) -> bool:
        with self._session._lock:
            return self._session._is_preparing_playback

    def is_resynthesizing(self) -> bool:
        with self._session._lock:
            return self._session._is_resynthesizing

    def rebuild_resynth(self) -> None:
        if self._session.audio_info is None:
            return
        with self._session._lock:
            self._session._is_resynthesizing = True
        self._session.synth.rebuild(self._session.store.all())
        with self._session._lock:
            self._session._is_resynthesizing = False
        self._session.player.load(self._mix_buffer(0.5), self._session.audio_info.sample_rate)
        self._session._playback_mode = None
        self.invalidate_cache()

    def update_mix_ratio(self, mix_ratio: float) -> bool:
        if self._session.audio_info is None:
            return False
        clamped_mix = float(np.clip(mix_ratio, 0.0, 1.0))
        self._session._last_mix_ratio = clamped_mix
        if self._session._playback_output_mode == "midi":
            return True
        if self._session._playback_mode != "normal":
            return False
        if not self._session.player.is_playing():
            return False
        speed_ratio = self._session._playback_speed_ratio
        time_stretch_mode = self._session._last_time_stretch_mode
        sample_rate = self._session.audio_info.sample_rate
        if abs(speed_ratio - 1.0) <= 1e-4:
            updated = self._mix_buffer(clamped_mix)
            self._session.player.update_buffer(updated)
            return True
        updated = self._build_speed_adjusted_buffer(clamped_mix, speed_ratio, sample_rate, time_stretch_mode)
        self._session.player.update_buffer(updated)
        cache_key = self._playback_cache_lookup_key(clamped_mix, speed_ratio, time_stretch_mode)
        self._store_playback_cache(cache_key, updated)
        return True

    def invalidate_cache(self) -> None:
        with self._session._lock:
            self._session._playback_content_revision += 1
            self._session._playback_cache_key = None
            self._session._playback_cache_buffer = None
            self._session._stretched_original_cache_key = None
            self._session._stretched_original_cache_buffer = None
            self._session._time_scaled_resynth_cache_key = None
            self._session._time_scaled_resynth_cache_buffer = None

    def render_cv_buffers(self, sample_rate: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        if self._session.audio_info is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.0, 1.0
        duration = self._session.audio_info.duration_sec
        total_samples = max(1, int(duration * sample_rate))
        freq_buffer = np.zeros(total_samples, dtype=np.float32)
        amp_buffer = np.zeros(total_samples, dtype=np.float32)
        amp_min = float("inf")
        amp_max = float("-inf")
        for partial in self._session.store.all():
            if partial.is_muted:
                continue
            points = partial.sorted_points()
            if len(points) < 2:
                continue
            for point in points:
                if point.amp < amp_min:
                    amp_min = point.amp
                if point.amp > amp_max:
                    amp_max = point.amp
            times = np.array([p.time for p in points], dtype=np.float64)
            freqs = np.array([p.freq for p in points], dtype=np.float64)
            amps = np.array([p.amp for p in points], dtype=np.float64)
            start = max(0.0, float(times[0]))
            end = min(float(times[-1]), duration)
            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)
            if end_idx <= start_idx:
                continue
            sample_times = (np.arange(end_idx - start_idx) / sample_rate) + start
            freq_interp = np.interp(sample_times, times, freqs)
            amp_interp = np.interp(sample_times, times, amps)
            current = amp_buffer[start_idx:end_idx]
            mask = amp_interp > current
            current[mask] = amp_interp[mask].astype(np.float32)
            freq_buffer[start_idx:end_idx][mask] = freq_interp[mask].astype(np.float32)
        if amp_min == float("inf") or amp_max == float("-inf"):
            amp_min, amp_max = 0.0, 1.0
        elif amp_max <= amp_min:
            amp_max = amp_min + 1.0
        return freq_buffer, amp_buffer, float(amp_min), float(amp_max)

    def _cancel_playback_prepare(self) -> None:
        with self._session._lock:
            self._session._playback_prepare_request_id += 1
            self._session._is_preparing_playback = False

    def _playback_cache_lookup_key(
        self,
        mix_ratio: float,
        speed_ratio: float,
        mode: str,
    ) -> tuple[int, float, float, str]:
        with self._session._lock:
            revision = self._session._playback_content_revision
        return (revision, round(float(mix_ratio), 4), round(float(speed_ratio), 6), mode)

    def _lookup_playback_cache(self, key: tuple[int, float, float, str]) -> np.ndarray | None:
        with self._session._lock:
            if self._session._playback_cache_key != key or self._session._playback_cache_buffer is None:
                return None
            return self._session._playback_cache_buffer

    def _store_playback_cache(self, key: tuple[int, float, float, str], buffer: np.ndarray) -> None:
        with self._session._lock:
            if key[0] != self._session._playback_content_revision:
                return
            self._session._playback_cache_key = key
            self._session._playback_cache_buffer = buffer.astype(np.float32, copy=False)

    def _mix_buffer(self, mix_ratio: float) -> np.ndarray:
        mix_ratio = float(np.clip(mix_ratio, 0.0, 1.0))
        resynth = self._session.synth.get_mix_buffer().astype(np.float32)
        if self._session.audio_data is None:
            return peak_normalize_buffer(resynth)
        original = self._match_length(self._session.audio_data, resynth.shape[0])
        mixed = (1.0 - mix_ratio) * original + mix_ratio * resynth
        return peak_normalize_buffer(mixed)

    def _match_length(self, audio: np.ndarray, length: int) -> np.ndarray:
        if audio.shape[0] == length:
            return audio.astype(np.float32)
        if audio.shape[0] < length:
            pad = np.zeros(length - audio.shape[0], dtype=np.float32)
            return np.concatenate([audio.astype(np.float32), pad])
        return audio[:length].astype(np.float32)

    def _build_speed_adjusted_buffer(
        self,
        mix_ratio: float,
        speed_ratio: float,
        sample_rate: int,
        time_stretch_mode: str,
    ) -> np.ndarray:
        clamped_mix = float(np.clip(mix_ratio, 0.0, 1.0))
        if clamped_mix >= 1.0 - 1e-6:
            return self._render_resynth_time_scaled_cached(speed_ratio, sample_rate)
        if self._session.audio_data is None:
            return self._render_resynth_time_scaled_cached(speed_ratio, sample_rate)
        if clamped_mix <= 1e-6:
            return self._render_original_time_stretched_cached(speed_ratio, sample_rate, time_stretch_mode)
        original = self._render_original_time_stretched_cached(speed_ratio, sample_rate, time_stretch_mode)
        resynth = self._render_resynth_time_scaled_cached(speed_ratio, sample_rate)
        length = max(original.shape[0], resynth.shape[0])
        original_matched = self._match_length(original, length)
        resynth_matched = self._match_length(resynth, length)
        mixed = (1.0 - clamped_mix) * original_matched + clamped_mix * resynth_matched
        return peak_normalize_buffer(mixed)

    def _render_original_time_stretched_cached(
        self,
        speed_ratio: float,
        sample_rate: int,
        time_stretch_mode: str,
    ) -> np.ndarray:
        source = self._session.audio_data
        if source is None:
            return np.zeros(1, dtype=np.float32)
        key = self._playback_stretch_component_key(speed_ratio, time_stretch_mode)
        with self._session._lock:
            if (
                self._session._stretched_original_cache_key == key
                and self._session._stretched_original_cache_buffer is not None
            ):
                return self._session._stretched_original_cache_buffer
        stretched = time_stretch_pitch_preserving(
            peak_normalize_buffer(source.astype(np.float32)),
            speed_ratio,
            sample_rate,
            mode=time_stretch_mode,
        )
        normalized = peak_normalize_buffer(stretched)
        with self._session._lock:
            if key[0] != self._session._playback_content_revision:
                return normalized
            self._session._stretched_original_cache_key = key
            self._session._stretched_original_cache_buffer = normalized.astype(np.float32, copy=False)
        return normalized

    def _render_resynth_time_scaled_cached(self, speed_ratio: float, sample_rate: int) -> np.ndarray:
        key = self._playback_resynth_component_key(speed_ratio, sample_rate)
        with self._session._lock:
            if (
                self._session._time_scaled_resynth_cache_key == key
                and self._session._time_scaled_resynth_cache_buffer is not None
            ):
                return self._session._time_scaled_resynth_cache_buffer
        rendered = self._render_resynth_time_scaled(speed_ratio, sample_rate)
        with self._session._lock:
            if key[0] != self._session._playback_content_revision:
                return rendered
            self._session._time_scaled_resynth_cache_key = key
            self._session._time_scaled_resynth_cache_buffer = rendered.astype(np.float32, copy=False)
        return rendered

    def _playback_stretch_component_key(self, speed_ratio: float, mode: str) -> tuple[int, float, str]:
        with self._session._lock:
            revision = self._session._playback_content_revision
        return (revision, round(float(speed_ratio), 6), mode)

    def _playback_resynth_component_key(self, speed_ratio: float, sample_rate: int) -> tuple[int, float, int]:
        with self._session._lock:
            revision = self._session._playback_content_revision
        return (revision, round(float(speed_ratio), 6), int(sample_rate))

    def _render_resynth_time_scaled(self, speed_ratio: float, sample_rate: int) -> np.ndarray:
        if self._session.audio_info is None:
            return np.zeros(1, dtype=np.float32)
        duration = self._session.audio_info.duration_sec
        total_samples = max(1, int((duration / speed_ratio) * sample_rate))
        buffer = np.zeros(total_samples, dtype=np.float64)
        fade_samples = max(1, int(sample_rate * 0.005))
        for partial in self._session.store.all():
            if partial.is_muted:
                continue
            points = partial.sorted_points()
            if len(points) < 2:
                continue
            times = np.array([point.time for point in points], dtype=np.float64)
            freqs = np.array([point.freq for point in points], dtype=np.float64)
            amps = np.array([point.amp for point in points], dtype=np.float64)
            start_scaled = max(0.0, float(times[0]) / speed_ratio)
            end_scaled = min(float(times[-1]) / speed_ratio, total_samples / sample_rate)
            start_idx = int(start_scaled * sample_rate)
            end_idx = int(end_scaled * sample_rate)
            if end_idx <= start_idx:
                continue
            scaled_times = (np.arange(end_idx - start_idx) / sample_rate) + start_scaled
            source_times = scaled_times * speed_ratio
            freq_interp = np.interp(source_times, times, freqs)
            amp_interp = np.interp(source_times, times, amps)
            phase = 2.0 * np.pi * np.cumsum(freq_interp) / sample_rate
            wave = np.sin(phase) * amp_interp
            fade = min(fade_samples, wave.size // 2)
            if fade > 0:
                fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float64)
                fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float64)
                wave[:fade] *= fade_in
                wave[-fade:] *= fade_out
            buffer[start_idx:end_idx] += wave
        return peak_normalize_buffer(buffer.astype(np.float32))

    def _harmonic_probe_state(self, time_sec: float) -> tuple[list[str], np.ndarray, np.ndarray]:
        partial_ids: list[str] = []
        freqs: list[float] = []
        amps: list[float] = []
        for partial in self._session.store.all():
            if partial.is_muted:
                continue
            sample = partial_sample_at_time(partial, time_sec)
            if sample is None:
                continue
            freq, amp = sample
            if freq <= 0.0 or amp <= 0.0:
                continue
            partial_ids.append(partial.id)
            freqs.append(freq)
            amps.append(amp)
        if not freqs:
            return [], np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        return partial_ids, np.asarray(freqs, dtype=np.float64), np.asarray(amps, dtype=np.float64)

    def harmonic_probe_tones(self, time_sec: float) -> tuple[np.ndarray, np.ndarray]:
        _, freqs, amps = self._harmonic_probe_state(time_sec)
        return freqs, amps

    def synth_mix_buffer(self) -> np.ndarray:
        return self._session.synth.get_mix_buffer().astype(np.float32)

    def partials(self) -> list[Partial]:
        return self._session.store.all()
