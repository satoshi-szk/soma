from __future__ import annotations

import numpy as np

from soma.synth import AudioPlayer


def test_probe_transition_preserves_phase_for_same_voice_id() -> None:
    player = AudioPlayer()
    player._sample_rate = 48000
    player._probe_freqs = np.asarray([440.0], dtype=np.float64)
    player._probe_amps = np.asarray([0.4], dtype=np.float64)
    player._probe_voice_ids = ["p1"]
    player._probe_phases = np.asarray([1.234], dtype=np.float64)
    player._probe_gain = 1.0

    player._begin_probe_transition(
        np.asarray([445.0], dtype=np.float64),
        np.asarray([0.6], dtype=np.float64),
        voice_ids=["p1"],
        stop_after_fade=False,
    )

    assert player._probe_phases.tolist() == [1.234]


def test_probe_transition_uses_new_phase_for_new_voice_id() -> None:
    player = AudioPlayer()
    player._sample_rate = 48000
    player._probe_freqs = np.asarray([440.0], dtype=np.float64)
    player._probe_amps = np.asarray([0.4], dtype=np.float64)
    player._probe_voice_ids = ["p1"]
    player._probe_phases = np.asarray([2.468], dtype=np.float64)
    player._probe_gain = 1.0

    player._begin_probe_transition(
        np.asarray([445.0], dtype=np.float64),
        np.asarray([0.6], dtype=np.float64),
        voice_ids=["p2"],
        stop_after_fade=False,
    )

    assert player._probe_phases.tolist() == [0.0]


def test_update_probe_same_voice_ids_does_not_restart_crossfade() -> None:
    player = AudioPlayer()
    player._playing = True
    player._mode = "probe"
    player._probe_voice_ids = ["p1"]
    player._probe_freqs = np.asarray([440.0], dtype=np.float64)
    player._probe_amps = np.asarray([0.4], dtype=np.float64)
    player._probe_crossfade_remaining = 7
    player._probe_prev_freqs = np.asarray([123.0], dtype=np.float64)
    player._probe_prev_amps = np.asarray([0.2], dtype=np.float64)

    ok = player.update_probe(
        np.asarray([445.0], dtype=np.float64),
        np.asarray([0.6], dtype=np.float64),
        voice_ids=["p1"],
    )

    assert ok is True
    assert player._probe_freqs.tolist() == [445.0]
    assert player._probe_amps.tolist() == [0.6]
    assert player._probe_crossfade_remaining == 7
    assert player._probe_prev_freqs.tolist() == [123.0]
