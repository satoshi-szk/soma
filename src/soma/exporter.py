from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from scipy.io import wavfile

from soma.models import Partial


@dataclass(frozen=True)
class MidiExportSettings:
    pitch_bend_range: int = 48
    amplitude_mapping: str = "velocity"  # velocity | pressure | cc74
    bpm: float = 120.0
    ticks_per_beat: int = 960


@dataclass(frozen=True)
class MpeExportSettings(MidiExportSettings):
    pass


@dataclass(frozen=True)
class MultiTrackExportSettings(MidiExportSettings):
    pass


@dataclass(frozen=True)
class MonophonicExportSettings(MidiExportSettings):
    pass


@dataclass(frozen=True)
class AudioExportSettings:
    sample_rate: int = 44100
    bit_depth: int = 16
    output_type: str = "sine"  # sine | cv


def export_mpe(
    partials: Iterable[Partial],
    output_path: Path,
    settings: MpeExportSettings,
) -> list[Path]:
    partial_list = [p for p in partials if p.points and not p.is_muted]
    if not partial_list:
        return []

    allocations = _allocate_voices(partial_list, max_voices=15)
    written: list[Path] = []
    base = output_path.with_suffix("")
    for index, group in enumerate(allocations, start=1):
        suffix = f"_{index:02d}.mid" if len(allocations) > 1 else ".mid"
        path = base.with_suffix("").with_name(base.name + suffix)
        midi = _build_mpe_file(group, settings)
        midi.save(path)
        written.append(path)
    return written


def export_multitrack_midi(
    partials: Iterable[Partial],
    output_path: Path,
    settings: MultiTrackExportSettings,
) -> list[Path]:
    partial_list = [p for p in partials if p.points and not p.is_muted]
    if not partial_list:
        return []

    allocations = _allocate_voices(partial_list, max_voices=None)
    written: list[Path] = []
    base = output_path.with_suffix("")
    for index, group in enumerate(allocations, start=1):
        suffix = f"_{index:02d}.mid" if len(allocations) > 1 else ".mid"
        path = base.with_suffix("").with_name(base.name + suffix)
        midi = _build_multitrack_file(group, settings)
        midi.save(path)
        written.append(path)
    return written


def export_monophonic_midi(
    partials: Iterable[Partial],
    output_path: Path,
    settings: MonophonicExportSettings,
) -> list[Path]:
    partial_list = [p for p in partials if p.points and not p.is_muted]
    if not partial_list:
        return []

    path = output_path.with_suffix("").with_suffix(".mid")
    midi = _build_monophonic_file(partial_list, settings)
    midi.save(path)
    return [path]


def export_audio(
    output_path: Path,
    audio_buffer: np.ndarray,
    settings: AudioExportSettings,
    freq_min: float,
    freq_max: float,
    pitch_buffer: np.ndarray | None = None,
    amp_buffer: np.ndarray | None = None,
) -> Path:
    if settings.output_type == "cv":
        if pitch_buffer is None or amp_buffer is None:
            raise ValueError("CV export requires pitch and amplitude buffers.")
        pitch_cv = _normalize_cv(pitch_buffer, freq_min, freq_max)
        amp_cv = np.clip(amp_buffer, 0.0, 1.0)
        stacked = np.vstack([pitch_cv, amp_cv]).T
        data = _convert_bit_depth(stacked, settings.bit_depth)
        wavfile.write(output_path, settings.sample_rate, data)
        return output_path

    normalized = np.clip(audio_buffer, -1.0, 1.0)
    data = _convert_bit_depth(normalized, settings.bit_depth)
    wavfile.write(output_path, settings.sample_rate, data)
    return output_path


def _build_mpe_file(voices: list[list[Partial]], settings: MpeExportSettings) -> MidiFile:
    midi = MidiFile(type=1, ticks_per_beat=settings.ticks_per_beat)
    tempo = bpm2tempo(settings.bpm)
    ticks_per_bar = settings.ticks_per_beat * 4

    events: list[TimedEvent] = []
    amp_min, amp_max = _amp_range([partial for voice in voices for partial in voice])
    for idx, voice in enumerate(voices):
        channel = idx + 1
        for partial in voice:
            events.extend(_partial_timed_events(partial, channel, settings, amp_min, amp_max))

    max_tick = _max_tick(events, settings.ticks_per_beat, tempo)
    rpn_events = _pitch_bend_rpn_tick_events(
        channels=list(range(16)),
        semitone_range=settings.pitch_bend_range,
        max_tick=max_tick,
        ticks_per_bar=ticks_per_bar,
    )

    master = _build_master_track(tempo, rpn_events)
    midi.tracks.append(master)

    note_track = _build_note_track(events, settings, tempo)
    midi.tracks.append(note_track)
    return midi


def _build_multitrack_file(voices: list[list[Partial]], settings: MultiTrackExportSettings) -> MidiFile:
    midi = MidiFile(type=1, ticks_per_beat=settings.ticks_per_beat)
    tempo = bpm2tempo(settings.bpm)
    ticks_per_bar = settings.ticks_per_beat * 4

    all_events: list[TimedEvent] = []
    amp_min, amp_max = _amp_range([partial for voice in voices for partial in voice])
    voice_events: list[list[TimedEvent]] = []
    for voice in voices:
        events: list[TimedEvent] = []
        for partial in voice:
            events.extend(_partial_timed_events(partial, 0, settings, amp_min, amp_max))
        voice_events.append(events)
        all_events.extend(events)

    max_tick = _max_tick(all_events, settings.ticks_per_beat, tempo)
    rpn_events = _pitch_bend_rpn_tick_events(
        channels=[0],
        semitone_range=settings.pitch_bend_range,
        max_tick=max_tick,
        ticks_per_bar=ticks_per_bar,
    )
    for idx, events in enumerate(voice_events):
        if idx == 0:
            midi.tracks.append(_build_note_track_with_rpn(events, settings, tempo, rpn_events))
        else:
            midi.tracks.append(_build_note_track(events, settings, tempo))
    return midi


def _build_monophonic_file(partials: list[Partial], settings: MonophonicExportSettings) -> MidiFile:
    midi = MidiFile(type=1, ticks_per_beat=settings.ticks_per_beat)
    tempo = bpm2tempo(settings.bpm)
    ticks_per_bar = settings.ticks_per_beat * 4

    amp_min, amp_max = _amp_range(partials)
    events: list[TimedEvent] = []
    for partial in partials:
        events.extend(_partial_timed_events(partial, 0, settings, amp_min, amp_max))

    intervals = {
        partial.id: (partial.sorted_points()[0].time, partial.sorted_points()[-1].time) for partial in partials
    }
    filtered = _filter_monophonic_events(events, intervals)

    max_tick = _max_tick(filtered, settings.ticks_per_beat, tempo)
    rpn_events = _pitch_bend_rpn_tick_events(
        channels=[0],
        semitone_range=settings.pitch_bend_range,
        max_tick=max_tick,
        ticks_per_bar=ticks_per_bar,
    )
    midi.tracks.append(_build_note_track_with_rpn(filtered, settings, tempo, rpn_events))
    return midi


@dataclass(frozen=True)
class TimedEvent:
    time: float
    order: int
    message: Message
    partial_id: str | None = None


def _partial_timed_events(
    partial: Partial,
    channel: int,
    settings: MidiExportSettings,
    amp_min: float,
    amp_max: float,
) -> list[TimedEvent]:
    points = sorted(partial.points, key=lambda p: p.time)
    if not points:
        return []
    events: list[TimedEvent] = []
    current_note = _freq_to_midi(points[0].freq)
    velocity = _amp_to_velocity(_normalize_amp(points[0].amp, amp_min, amp_max))
    events.append(
        TimedEvent(
            time=points[0].time,
            order=_message_order("note_on"),
            message=Message("note_on", note=current_note, velocity=velocity, channel=channel),
            partial_id=partial.id,
        )
    )

    for point in points:
        if _needs_retrigger(point.freq, current_note, settings.pitch_bend_range):
            events.append(
                TimedEvent(
                    time=point.time,
                    order=_message_order("note_off"),
                    message=Message("note_off", note=current_note, velocity=0, channel=channel),
                    partial_id=partial.id,
                )
            )
            current_note = _freq_to_midi(point.freq)
            velocity = _amp_to_velocity(_normalize_amp(point.amp, amp_min, amp_max))
            events.append(
                TimedEvent(
                    time=point.time,
                    order=_message_order("note_on"),
                    message=Message("note_on", note=current_note, velocity=velocity, channel=channel),
                    partial_id=partial.id,
                )
            )
        pitch_bend = _freq_to_pitch_bend(point.freq, current_note, settings.pitch_bend_range)
        normalized_amp = _normalize_amp(point.amp, amp_min, amp_max)
        events.append(
            TimedEvent(
                time=point.time,
                order=_message_order("pitchwheel"),
                message=Message("pitchwheel", pitch=pitch_bend, channel=channel),
                partial_id=partial.id,
            )
        )
        if settings.amplitude_mapping == "pressure":
            events.append(
                TimedEvent(
                    time=point.time,
                    order=_message_order("aftertouch"),
                    message=Message("aftertouch", value=_amp_to_cc(normalized_amp), channel=channel),
                    partial_id=partial.id,
                )
            )
        elif settings.amplitude_mapping == "cc74":
            events.append(
                TimedEvent(
                    time=point.time,
                    order=_message_order("control_change"),
                    message=Message("control_change", control=74, value=_amp_to_cc(normalized_amp), channel=channel),
                    partial_id=partial.id,
                )
            )
    events.append(
        TimedEvent(
            time=points[-1].time,
            order=_message_order("note_off"),
            message=Message("note_off", note=current_note, velocity=0, channel=channel),
            partial_id=partial.id,
        )
    )
    return events


def _allocate_voices(partials: list[Partial], max_voices: int | None) -> list[list[list[Partial]]]:
    groups: list[list[list[Partial]]] = []
    voice_end_times: list[float] = []
    voices: list[list[Partial]] = []

    def start_new_group(partial: Partial, end_time: float) -> None:
        nonlocal voices, voice_end_times
        voices = [[partial]]
        voice_end_times = [end_time]
        groups.append(voices)

    for partial in sorted(partials, key=lambda p: p.points[0].time):
        start = partial.points[0].time
        end = partial.points[-1].time
        assigned = False
        for idx, end_time in enumerate(voice_end_times):
            if start >= end_time:
                voice_end_times[idx] = end
                voices[idx].append(partial)
                assigned = True
                break
        if assigned:
            continue

        if max_voices is None or len(voice_end_times) < max_voices:
            voice_end_times.append(end)
            voices.append([partial])
            if not groups:
                groups.append(voices)
        else:
            start_new_group(partial, end)

    if not groups and voices:
        groups.append(voices)
    return groups


def _pitch_bend_rpn_messages(channel: int, semitone_range: int) -> list[Message]:
    """Generate RPN messages for pitch bend range setting.

    Sends the sequence twice (2 ticks) to handle DAW loop playback
    that may skip the first tick. Ends with RPN NULL to prevent
    subsequent Data Entry from being misinterpreted.
    """
    msb = min(127, max(0, semitone_range))
    messages: list[Message] = []
    for _ in range(2):
        messages.extend([
            Message("control_change", control=101, value=0, channel=channel),
            Message("control_change", control=100, value=0, channel=channel),
            Message("control_change", control=6, value=msb, channel=channel),
            Message("control_change", control=38, value=0, channel=channel),
        ])
    # RPN NULL to terminate the RPN sequence
    messages.extend([
        Message("control_change", control=101, value=127, channel=channel),
        Message("control_change", control=100, value=127, channel=channel),
    ])
    return messages


def _pitch_bend_rpn_tick_events(
    channels: list[int],
    semitone_range: int,
    max_tick: int,
    ticks_per_bar: int,
) -> list[tuple[int, int, Message]]:
    events: list[tuple[int, int, Message]] = []
    bar_tick = 0
    while bar_tick <= max_tick:
        for channel in channels:
            for msg in _pitch_bend_rpn_messages(channel, semitone_range):
                events.append((bar_tick, 0, msg.copy()))
        bar_tick += ticks_per_bar
    return events


def _build_master_track(tempo: int, rpn_events: list[tuple[int, int, Message]]) -> MidiTrack:
    track = MidiTrack()
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    last_tick = 0
    for tick, _, message in sorted(rpn_events, key=lambda item: (item[0], item[1])):
        delta = max(0, tick - last_tick)
        message.time = delta
        track.append(message)
        last_tick = tick
    return track


def _build_note_track(events: list[TimedEvent], settings: MidiExportSettings, tempo: int) -> MidiTrack:
    tick_events = _note_tick_events(events, settings, tempo)
    return _build_track_from_tick_events(tick_events)


def _build_note_track_with_rpn(
    events: list[TimedEvent],
    settings: MidiExportSettings,
    tempo: int,
    rpn_events: list[tuple[int, int, Message]],
) -> MidiTrack:
    tick_events = _note_tick_events(events, settings, tempo)
    combined = rpn_events + tick_events
    track = MidiTrack()
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    last_tick = 0
    for tick, _, message in sorted(combined, key=lambda item: (item[0], item[1])):
        delta = max(0, tick - last_tick)
        message.time = delta
        track.append(message)
        last_tick = tick
    return track


def _note_tick_events(
    events: list[TimedEvent],
    settings: MidiExportSettings,
    tempo: int,
) -> list[tuple[int, int, Message]]:
    tick_events: list[tuple[int, int, Message]] = []
    for event in events:
        tick = int(second2tick(event.time, settings.ticks_per_beat, tempo))
        tick_events.append((tick, event.order, event.message))
    return tick_events


def _build_track_from_tick_events(tick_events: list[tuple[int, int, Message]]) -> MidiTrack:
    tick_events.sort(key=lambda item: (item[0], item[1]))
    track = MidiTrack()
    last_tick = 0
    for tick, _, message in tick_events:
        delta = max(0, tick - last_tick)
        message.time = delta
        track.append(message)
        last_tick = tick
    return track


def _filter_monophonic_events(
    events: list[TimedEvent],
    intervals: dict[str, tuple[float, float]],
) -> list[TimedEvent]:
    def active_partial_id(time: float) -> str | None:
        active_id: str | None = None
        active_start = float("-inf")
        for partial_id, (start, end) in intervals.items():
            if start <= time <= end and start >= active_start:
                active_id = partial_id
                active_start = start
        return active_id

    filtered: list[TimedEvent] = []
    for event in events:
        if event.partial_id is None:
            filtered.append(event)
            continue
        if event.message.type in {"note_on", "note_off"}:
            filtered.append(event)
            continue
        if event.partial_id == active_partial_id(event.time):
            filtered.append(event)
    return filtered


def _max_tick(events: list[TimedEvent], ticks_per_beat: int, tempo: int) -> int:
    max_time_sec = max((event.time for event in events), default=0.0)
    return int(second2tick(max_time_sec, ticks_per_beat, tempo))


def _message_order(message_type: str) -> int:
    if message_type == "note_off":
        return 1
    if message_type == "note_on":
        return 2
    if message_type == "pitchwheel":
        return 3
    if message_type in {"aftertouch", "control_change"}:
        return 4
    return 5


def _freq_to_midi(freq: float) -> int:
    return int(np.clip(np.round(69 + 12 * np.log2(freq / 440.0)), 0, 127))


def _amp_range(partials: list[Partial]) -> tuple[float, float]:
    min_amp = float("inf")
    max_amp = float("-inf")
    for partial in partials:
        for point in partial.points:
            if point.amp < min_amp:
                min_amp = point.amp
            if point.amp > max_amp:
                max_amp = point.amp
    if min_amp == float("inf") or max_amp == float("-inf"):
        return 0.0, 1.0
    if max_amp <= min_amp:
        return min_amp, min_amp + 1.0
    return min_amp, max_amp


def _normalize_amp(amp: float, amp_min: float, amp_max: float) -> float:
    if amp_max <= amp_min:
        return 1.0
    return float(np.clip((amp - amp_min) / (amp_max - amp_min), 0.0, 1.0))


def _needs_retrigger(freq: float, midi_note: int, pitch_bend_range: int) -> bool:
    offset = _freq_to_semitone_offset(freq, midi_note)
    return abs(offset) > pitch_bend_range


def _freq_to_semitone_offset(freq: float, midi_note: int) -> float:
    target = 69 + 12 * np.log2(freq / 440.0)
    return float(target - midi_note)


def _freq_to_pitch_bend(freq: float, midi_note: int, pitch_bend_range: int) -> int:
    offset = _freq_to_semitone_offset(freq, midi_note)
    normalized = np.clip(offset / pitch_bend_range, -1.0, 1.0)
    return int(normalized * 8191)


def _amp_to_velocity(amp: float) -> int:
    return int(np.clip(round(1 + amp * 126), 1, 127))


def _amp_to_cc(amp: float) -> int:
    return int(np.clip(round(1 + amp * 126), 1, 127))


def _convert_bit_depth(data: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth == 24:
        max_val = 2**23 - 1
        return np.asarray(np.clip(data, -1.0, 1.0) * max_val, dtype=np.int32)
    if bit_depth == 32:
        return np.asarray(np.clip(data, -1.0, 1.0), dtype=np.float32)
    max_val = 2**15 - 1
    return np.asarray(np.clip(data, -1.0, 1.0) * max_val, dtype=np.int16)


def _normalize_cv(buffer: np.ndarray, freq_min: float, freq_max: float) -> np.ndarray:
    if buffer.size == 0:
        return buffer.astype(np.float32)
    log_min = np.log2(max(freq_min, 1.0))
    log_max = np.log2(max(freq_max, freq_min + 1.0))
    normalized = (np.log2(np.clip(buffer, freq_min, freq_max)) - log_min) / max(1e-6, log_max - log_min)
    return np.asarray(np.clip(normalized * 2.0 - 1.0, -1.0, 1.0), dtype=np.float32)
