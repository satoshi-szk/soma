from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
from scipy.io import wavfile

from soma.models import Partial


@dataclass(frozen=True)
class MpeExportSettings:
    pitch_bend_range: int = 48
    amplitude_mapping: str = "velocity"  # velocity | pressure | cc74
    bpm: float = 120.0
    ticks_per_beat: int = 960


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

    allocations = _allocate_channels(partial_list, max_channels=15)
    written: list[Path] = []
    base = output_path.with_suffix("")
    for index, group in enumerate(allocations, start=1):
        suffix = f"_{index:02d}.mid" if len(allocations) > 1 else ".mid"
        path = base.with_suffix("").with_name(base.name + suffix)
        midi = _build_mpe_file(group, settings)
        midi.save(path)
        written.append(path)
    return written


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


def _build_mpe_file(partials: list[Partial], settings: MpeExportSettings) -> MidiFile:
    midi = MidiFile(ticks_per_beat=settings.ticks_per_beat)
    tempo = bpm2tempo(settings.bpm)

    master = MidiTrack()
    master.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    master.extend(_pitch_bend_rpn_messages(0, settings.pitch_bend_range))
    midi.tracks.append(master)

    events: list[tuple[float, Message]] = []
    for idx, partial in enumerate(partials):
        channel = (idx % 15) + 1
        events.extend(_partial_events(partial, channel, settings))

    track = MidiTrack()
    track.extend(_pitch_bend_rpn_messages(0, settings.pitch_bend_range))
    for channel in range(1, 16):
        track.extend(_pitch_bend_rpn_messages(channel, settings.pitch_bend_range))

    events.sort(key=lambda item: item[0])
    last_tick = 0
    for time_sec, message in events:
        tick = int(second2tick(time_sec, settings.ticks_per_beat, tempo))
        delta = max(0, tick - last_tick)
        message.time = delta
        track.append(message)
        last_tick = tick
    midi.tracks.append(track)
    return midi


def _partial_events(partial: Partial, channel: int, settings: MpeExportSettings) -> list[tuple[float, Message]]:
    points = sorted(partial.points, key=lambda p: p.time)
    if not points:
        return []
    events: list[tuple[float, Message]] = []
    current_note = _freq_to_midi(points[0].freq)
    velocity = _amp_to_velocity(points[0].amp)
    events.append((points[0].time, Message("note_on", note=current_note, velocity=velocity, channel=channel)))

    for point in points:
        if _needs_retrigger(point.freq, current_note, settings.pitch_bend_range):
            events.append((point.time, Message("note_off", note=current_note, velocity=0, channel=channel)))
            current_note = _freq_to_midi(point.freq)
            velocity = _amp_to_velocity(point.amp)
            events.append((point.time, Message("note_on", note=current_note, velocity=velocity, channel=channel)))
        pitch_bend = _freq_to_pitch_bend(point.freq, current_note, settings.pitch_bend_range)
        events.append((point.time, Message("pitchwheel", pitch=pitch_bend, channel=channel)))
        if settings.amplitude_mapping == "pressure":
            events.append((point.time, Message("aftertouch", value=_amp_to_cc(point.amp), channel=channel)))
        elif settings.amplitude_mapping == "cc74":
            events.append(
                (point.time, Message("control_change", control=74, value=_amp_to_cc(point.amp), channel=channel))
            )
    events.append((points[-1].time, Message("note_off", note=current_note, velocity=0, channel=channel)))
    return events


def _allocate_channels(partials: list[Partial], max_channels: int) -> list[list[Partial]]:
    groups: list[list[Partial]] = [[]]
    channel_end_times: list[float] = []

    for partial in sorted(partials, key=lambda p: p.points[0].time):
        start = partial.points[0].time
        end = partial.points[-1].time
        assigned = False
        for idx, end_time in enumerate(channel_end_times):
            if start >= end_time:
                channel_end_times[idx] = end
                groups[-1].append(partial)
                assigned = True
                break
        if assigned:
            continue

        if len(channel_end_times) < max_channels:
            channel_end_times.append(end)
            groups[-1].append(partial)
        else:
            groups.append([partial])
            channel_end_times = [end]

    return groups


def _pitch_bend_rpn_messages(channel: int, semitone_range: int) -> list[Message]:
    msb = min(127, max(0, semitone_range))
    return [
        Message("control_change", control=101, value=0, channel=channel),
        Message("control_change", control=100, value=0, channel=channel),
        Message("control_change", control=6, value=msb, channel=channel),
        Message("control_change", control=38, value=0, channel=channel),
    ]


def _freq_to_midi(freq: float) -> int:
    return int(np.clip(np.round(69 + 12 * np.log2(freq / 440.0)), 0, 127))


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
    return int(np.clip(_amp_to_midi_db(amp), 1, 127))


def _amp_to_cc(amp: float) -> int:
    return int(np.clip(_amp_to_midi_db(amp), 0, 127))


def _amp_to_midi_db(amp: float, min_db: float = -60.0) -> int:
    db = min_db if amp <= 0 else max(min_db, 20.0 * np.log10(amp))
    normalized = (db - min_db) / max(1e-6, -min_db)
    return int(round(np.clip(normalized, 0.0, 1.0) * 127))


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
