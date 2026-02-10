from __future__ import annotations

import threading
import time
from typing import Any

import mido
from mido import Message, MidiFile, bpm2tempo, merge_tracks, tick2second


class MidiPlayer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._playing = False
        self._position_sec = 0.0

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
        with self._lock:
            self._playing = False

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def position_sec(self) -> float:
        with self._lock:
            return self._position_sec

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
