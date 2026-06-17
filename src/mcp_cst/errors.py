"""Structured error codes returned via MCP tool-error mechanism."""

from __future__ import annotations
from enum import StrEnum


class ErrorCode(StrEnum):
    TICKET_NOT_FOUND = "TICKET_NOT_FOUND"
    UNSUPPORTED_GROUP_BY = "UNSUPPORTED_GROUP_BY"
    UNSUPPORTED_FILTER = "UNSUPPORTED_FILTER"
    NO_GROUNDING_AVAILABLE = "NO_GROUNDING_AVAILABLE"
    INJECTION_DETECTED = "INJECTION_DETECTED"
    DATASET_UNAVAILABLE = "DATASET_UNAVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"


class McpCstError(Exception):
    """Raised by tool/resource/prompt code for any structured error."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def to_payload(err: McpCstError) -> dict:
    """Render an error as the JSON payload returned to the MCP client."""
    return {"error": {"code": err.code.value, "message": err.message}}
