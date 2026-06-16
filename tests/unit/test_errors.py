import json
import pytest
from mcp_cst.errors import McpCstError, ErrorCode, to_payload


def test_error_codes_defined():
    for code in [
        "TICKET_NOT_FOUND",
        "UNSUPPORTED_GROUP_BY",
        "UNSUPPORTED_FILTER",
        "NO_GROUNDING_AVAILABLE",
        "INJECTION_DETECTED",
        "NO_LLM_CONFIGURED",
        "DATASET_UNAVAILABLE",
    ]:
        assert hasattr(ErrorCode, code)


def test_to_payload_shape():
    err = McpCstError(ErrorCode.TICKET_NOT_FOUND, "no such ticket: abc")
    payload = to_payload(err)
    assert payload == {"error": {"code": "TICKET_NOT_FOUND", "message": "no such ticket: abc"}}
    # round-trips through JSON
    assert json.loads(json.dumps(payload)) == payload


def test_raises_with_code():
    with pytest.raises(McpCstError) as exc:
        raise McpCstError(ErrorCode.INJECTION_DETECTED, "found 'ignore previous instructions'")
    assert exc.value.code == ErrorCode.INJECTION_DETECTED
    assert "ignore previous" in str(exc.value)
