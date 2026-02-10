from __future__ import annotations

from mido import Message

from soma.midi_player import MidiPlayer


class _FakeOutput:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.closed = False

    def send(self, message: Message) -> None:
        self.messages.append(message.copy())

    def close(self) -> None:
        self.closed = True


def test_probe_update_keeps_legato_for_same_partial(monkeypatch) -> None:
    player = MidiPlayer()
    output = _FakeOutput()
    monkeypatch.setattr("soma.midi_player.mido.open_output", lambda _: output)

    ok = player.start_probe(
        output_name="dummy",
        partial_ids=["p1"],
        freqs=[440.0],
        amps=[0.5],
        pitch_bend_range=48,
        midi_mode="mpe",
        amplitude_mapping="pressure",
        amplitude_curve="linear",
    )
    assert ok is True

    start_len = len(output.messages)
    updated = player.update_probe(
        partial_ids=["p1"],
        freqs=[446.0],
        amps=[0.6],
        midi_mode="mpe",
        amplitude_mapping="pressure",
        amplitude_curve="linear",
    )
    assert updated is True

    delta = output.messages[start_len:]
    assert all(msg.type != "note_off" for msg in delta)
    assert all(msg.type != "note_on" for msg in delta)
    assert any(msg.type == "pitchwheel" for msg in delta)
    assert any(msg.type == "aftertouch" for msg in delta)


def test_probe_update_sends_note_off_when_partial_disappears(monkeypatch) -> None:
    player = MidiPlayer()
    output = _FakeOutput()
    monkeypatch.setattr("soma.midi_player.mido.open_output", lambda _: output)

    ok = player.start_probe(
        output_name="dummy",
        partial_ids=["p1"],
        freqs=[440.0],
        amps=[0.5],
        pitch_bend_range=48,
        midi_mode="mpe",
        amplitude_mapping="pressure",
        amplitude_curve="linear",
    )
    assert ok is True

    start_len = len(output.messages)
    updated = player.update_probe(
        partial_ids=[],
        freqs=[],
        amps=[],
        midi_mode="mpe",
        amplitude_mapping="pressure",
        amplitude_curve="linear",
    )
    assert updated is True

    delta = output.messages[start_len:]
    assert any(msg.type == "note_off" for msg in delta)


def test_probe_mpe_mode_uses_mpe_note_channels(monkeypatch) -> None:
    player = MidiPlayer()
    output = _FakeOutput()
    monkeypatch.setattr("soma.midi_player.mido.open_output", lambda _: output)

    ok = player.start_probe(
        output_name="dummy",
        partial_ids=["p1"],
        freqs=[440.0],
        amps=[0.5],
        pitch_bend_range=48,
        midi_mode="mpe",
        amplitude_mapping="pressure",
        amplitude_curve="linear",
    )
    assert ok is True

    note_on_messages = [msg for msg in output.messages if msg.type == "note_on"]
    assert len(note_on_messages) == 1
    # MPE lower-zone note channel should start from MIDI ch2 (0-based channel=1).
    assert note_on_messages[0].channel == 1


def test_probe_cc74_mapping_sends_cc74_not_aftertouch(monkeypatch) -> None:
    player = MidiPlayer()
    output = _FakeOutput()
    monkeypatch.setattr("soma.midi_player.mido.open_output", lambda _: output)

    ok = player.start_probe(
        output_name="dummy",
        partial_ids=["p1"],
        freqs=[440.0],
        amps=[0.5],
        pitch_bend_range=48,
        midi_mode="mpe",
        amplitude_mapping="cc74",
        amplitude_curve="linear",
    )
    assert ok is True

    delta_start = len(output.messages)
    updated = player.update_probe(
        partial_ids=["p1"],
        freqs=[442.0],
        amps=[0.6],
        midi_mode="mpe",
        amplitude_mapping="cc74",
        amplitude_curve="linear",
    )
    assert updated is True
    delta = output.messages[delta_start:]
    assert any(msg.type == "control_change" and msg.control == 74 for msg in delta)
    assert all(msg.type != "aftertouch" for msg in delta)
