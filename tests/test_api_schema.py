from soma.api_schema import TracePartialPayload, UpdateSettingsPayload, parse_payload


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
