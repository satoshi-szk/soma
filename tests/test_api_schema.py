from soma.api_schema import (
    ExportMpePayload,
    OpenAudioDataPayload,
    OpenAudioPathPayload,
    PlayPayload,
    TracePartialPayload,
    UpdateSettingsPayload,
    parse_payload,
)


def test_parse_payload_rejects_non_dict() -> None:
    parsed, error = parse_payload(UpdateSettingsPayload, ["not", "a", "dict"])
    assert parsed is None
    assert error is not None
    assert "expected object" in error


def test_parse_payload_accepts_trace() -> None:
    parsed, error = parse_payload(TracePartialPayload, {"trace": [[0.0, 440.0], [1.0, 880.0]]})
    assert error is None
    assert parsed is not None
    assert parsed.trace == [(0.0, 440.0), (1.0, 880.0)]


def test_parse_payload_rejects_bad_trace() -> None:
    parsed, error = parse_payload(TracePartialPayload, {"trace": [[1.0]]})
    assert parsed is None
    assert error is not None


def test_parse_payload_rejects_non_monotonic_trace() -> None:
    parsed, error = parse_payload(
        TracePartialPayload,
        {"trace": [[0.0, 440.0], [0.2, 450.0], [0.1, 460.0]]},
    )
    assert parsed is None
    assert error is not None
    assert "one time direction" in error


def test_parse_payload_accepts_audio_path() -> None:
    parsed, error = parse_payload(OpenAudioPathPayload, {"path": "/tmp/example.wav"})
    assert error is None
    assert parsed is not None
    assert parsed.path == "/tmp/example.wav"


def test_parse_payload_accepts_audio_data() -> None:
    parsed, error = parse_payload(OpenAudioDataPayload, {"name": "example.wav", "data_base64": "AAAA"})
    assert error is None
    assert parsed is not None
    assert parsed.name == "example.wav"


def test_parse_payload_accepts_play_speed_ratio() -> None:
    parsed, error = parse_payload(PlayPayload, {"mix_ratio": 0.4, "speed_ratio": 0.125})
    assert error is None
    assert parsed is not None
    assert parsed.speed_ratio == 0.125


def test_parse_payload_rejects_play_speed_ratio_out_of_range() -> None:
    parsed, error = parse_payload(PlayPayload, {"speed_ratio": 8.1})
    assert parsed is None
    assert error is not None


def test_parse_payload_accepts_play_time_stretch_mode() -> None:
    parsed, error = parse_payload(PlayPayload, {"time_stretch_mode": "native"})
    assert error is None
    assert parsed is not None
    assert parsed.time_stretch_mode == "native"


def test_parse_payload_rejects_play_invalid_time_stretch_mode() -> None:
    parsed, error = parse_payload(PlayPayload, {"time_stretch_mode": "invalid"})
    assert parsed is None
    assert error is not None


def test_parse_payload_accepts_export_mpe_cc_update_rate() -> None:
    parsed, error = parse_payload(ExportMpePayload, {"cc_update_rate_hz": 400})
    assert error is None
    assert parsed is not None
    assert parsed.cc_update_rate_hz == 400
