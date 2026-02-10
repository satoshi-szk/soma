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
    )
    assert ok is True

    start_len = len(output.messages)
    updated = player.update_probe(partial_ids=["p1"], freqs=[446.0], amps=[0.6])
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
    )
    assert ok is True

    start_len = len(output.messages)
    updated = player.update_probe(partial_ids=[], freqs=[], amps=[])
    assert updated is True

    delta = output.messages[start_len:]
    assert any(msg.type == "note_off" for msg in delta)
