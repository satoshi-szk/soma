from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import mido
import numpy as np
from mido import Message, MidiFile, bpm2tempo, merge_tracks, tick2second


@dataclass
class _ProbeVoice:
    channel: int
    note: int


class MidiPlayer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._playing = False
        self._position_sec = 0.0
        self._probe_output: Any | None = None
        self._probe_pitch_bend_range = 48
        self._probe_voices: dict[str, _ProbeVoice] = {}
        self._probe_free_channels: set[int] = set(range(16))

    def list_outputs(self) -> list[str]:
        try:
            return list(mido.get_output_names())
        except Exception:
            return []

    def play(self, midi: MidiFile, output_name: str, start_position_sec: float, speed_ratio: float) -> None:
        self.play_until(
            midi=midi,
            output_name=output_name,
            start_position_sec=start_position_sec,
            speed_ratio=speed_ratio,
            end_position_sec=None,
        )

    def play_until(
        self,
        midi: MidiFile,
        output_name: str,
        start_position_sec: float,
        speed_ratio: float,
        end_position_sec: float | None,
    ) -> None:
        clamped_start = max(0.0, float(start_position_sec))
        clamped_speed = max(0.125, min(8.0, float(speed_ratio)))
        target_end_position = None if end_position_sec is None else max(clamped_start, float(end_position_sec))
        events = self._events_from_midi(
            midi=midi,
            start_position_sec=clamped_start,
            speed_ratio=clamped_speed,
        )
        self.stop()
        self._stop_event.clear()
        with self._lock:
            self._position_sec = clamped_start
            self._playing = True

        def _worker() -> None:
            output = None
            start_wall = time.monotonic()
            channels = {msg.channel for _, msg in events if hasattr(msg, "channel")}
            if not channels:
                channels = set(range(16))
            try:
                output = mido.open_output(output_name)
                for offset_sec, message in events:
                    wait_sec = offset_sec - (time.monotonic() - start_wall)
                    if wait_sec > 0 and self._stop_event.wait(wait_sec):
                        break
                    if self._stop_event.is_set():
                        break
                    output.send(message)
                    with self._lock:
                        self._position_sec = max(0.0, clamped_start + offset_sec * clamped_speed)

                if target_end_position is not None:
                    intended_duration = max(0.0, target_end_position - clamped_start) / clamped_speed
                    elapsed = time.monotonic() - start_wall
                    deadline = time.monotonic() + max(0.0, intended_duration - elapsed)
                    # 末尾の無音区間でも playhead を継続更新する。
                    while not self._stop_event.is_set():
                        now = time.monotonic()
                        if now >= deadline:
                            break
                        elapsed = now - start_wall
                        with self._lock:
                            self._position_sec = min(target_end_position, clamped_start + elapsed * clamped_speed)
                        self._stop_event.wait(min(0.03, max(0.0, deadline - now)))
                with self._lock:
                    if target_end_position is not None:
                        self._position_sec = target_end_position
            finally:
                if output is not None:
                    self._send_panic(output, channels)
                    output.close()
                with self._lock:
                    self._playing = False

        thread = threading.Thread(target=_worker, name="soma-midi-playback", daemon=True)
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._stop_probe_output()
        with self._lock:
            self._playing = False

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def position_sec(self) -> float:
        with self._lock:
            return self._position_sec

    def start_probe(
        self,
        output_name: str,
        partial_ids: list[str],
        freqs: list[float],
        amps: list[float],
        pitch_bend_range: int,
    ) -> bool:
        self.stop()
        try:
            output = mido.open_output(output_name)
        except Exception:
            return False
        self._probe_output = output
        self._probe_voices = {}
        self._probe_free_channels = set(range(16))
        self._probe_pitch_bend_range = max(1, min(96, int(pitch_bend_range)))
        self._apply_probe_state(partial_ids, freqs, amps)
        with self._lock:
            self._playing = True
        return True

    def update_probe(self, partial_ids: list[str], freqs: list[float], amps: list[float]) -> bool:
        if self._probe_output is None:
            return False
        self._apply_probe_state(partial_ids, freqs, amps)
        return True

    def stop_probe(self) -> None:
        self._stop_probe_output()
        with self._lock:
            self._playing = False

    def _stop_probe_output(self) -> None:
        output = self._probe_output
        if output is None:
            return
        for voice in self._probe_voices.values():
            output.send(Message("note_off", channel=voice.channel, note=voice.note, velocity=0))
        self._send_panic(output, set(range(16)))
        output.close()
        self._probe_voices = {}
        self._probe_free_channels = set(range(16))
        self._probe_output = None

    def _apply_probe_state(self, partial_ids: list[str], freqs: list[float], amps: list[float]) -> None:
        output = self._probe_output
        if output is None:
            return
        incoming: list[tuple[str, float, float]] = []
        max_voice = min(len(partial_ids), len(freqs), len(amps), 16)
        for index in range(max_voice):
            partial_id = partial_ids[index]
            freq = float(freqs[index])
            if not np.isfinite(freq) or freq <= 0:
                continue
            amp = float(np.clip(amps[index], 0.0, 1.0))
            incoming.append((partial_id, freq, amp))

        incoming_ids = {partial_id for partial_id, _, _ in incoming}
        for stale_id in list(self._probe_voices.keys()):
            if stale_id in incoming_ids:
                continue
            stale = self._probe_voices.pop(stale_id)
            output.send(Message("note_off", channel=stale.channel, note=stale.note, velocity=0))
            self._probe_free_channels.add(stale.channel)

        for partial_id, freq, amp in incoming:
            voice = self._probe_voices.get(partial_id)
            if voice is None:
                if not self._probe_free_channels:
                    continue
                channel = min(self._probe_free_channels)
                self._probe_free_channels.remove(channel)
                self._send_pitch_bend_range(channel, self._probe_pitch_bend_range)
                note = self._freq_to_midi(freq)
                velocity = self._amp_to_value(amp)
                output.send(Message("note_on", channel=channel, note=note, velocity=velocity))
                voice = _ProbeVoice(channel=channel, note=note)
                self._probe_voices[partial_id] = voice
            elif self._needs_retrigger(freq, voice.note, self._probe_pitch_bend_range):
                output.send(Message("note_off", channel=voice.channel, note=voice.note, velocity=0))
                note = self._freq_to_midi(freq)
                velocity = self._amp_to_value(amp)
                output.send(Message("note_on", channel=voice.channel, note=note, velocity=velocity))
                voice.note = note
            output.send(Message("pitchwheel", channel=voice.channel, pitch=self._freq_to_pitch_bend(freq, voice.note)))
            output.send(Message("aftertouch", channel=voice.channel, value=self._amp_to_value(amp)))

    def _send_pitch_bend_range(self, channel: int, semitone_range: int) -> None:
        output = self._probe_output
        if output is None:
            return
        output.send(Message("control_change", channel=channel, control=101, value=0))
        output.send(Message("control_change", channel=channel, control=100, value=0))
        output.send(Message("control_change", channel=channel, control=6, value=int(np.clip(semitone_range, 1, 96))))
        output.send(Message("control_change", channel=channel, control=38, value=0))
        output.send(Message("control_change", channel=channel, control=101, value=127))
        output.send(Message("control_change", channel=channel, control=100, value=127))

    def _freq_to_midi(self, freq: float) -> int:
        return int(np.clip(np.round(69 + 12 * np.log2(freq / 440.0)), 0, 127))

    def _amp_to_value(self, amp: float) -> int:
        return int(np.clip(round(1 + amp * 126), 1, 127))

    def _needs_retrigger(self, freq: float, midi_note: int, pitch_bend_range: int) -> bool:
        offset = self._freq_to_semitone_offset(freq, midi_note)
        return abs(offset) > pitch_bend_range

    def _freq_to_semitone_offset(self, freq: float, midi_note: int) -> float:
        target = 69 + 12 * np.log2(freq / 440.0)
        return float(target - midi_note)

    def _freq_to_pitch_bend(self, freq: float, midi_note: int) -> int:
        offset = self._freq_to_semitone_offset(freq, midi_note)
        normalized = np.clip(offset / self._probe_pitch_bend_range, -1.0, 1.0)
        return int(normalized * 8191)

    def _send_panic(self, output: Any, channels: set[int]) -> None:
        for channel in sorted(channels):
            output.send(Message("control_change", channel=channel, control=123, value=0))
            output.send(Message("control_change", channel=channel, control=120, value=0))

    def _events_from_midi(
        self,
        midi: MidiFile,
        start_position_sec: float,
        speed_ratio: float,
    ) -> list[tuple[float, Message]]:
        current_sec = 0.0
        tempo = bpm2tempo(120.0)
        events: list[tuple[float, Message]] = []
        for message in merge_tracks(midi.tracks):
            current_sec += tick2second(message.time, midi.ticks_per_beat, tempo)
            if message.type == "set_tempo":
                tempo = message.tempo
                continue
            if message.is_meta:
                continue
            if current_sec < start_position_sec:
                continue
            offset_sec = (current_sec - start_position_sec) / speed_ratio
            events.append((offset_sec, message.copy(time=0)))
        return events
