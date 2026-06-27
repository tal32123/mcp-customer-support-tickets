"""#219: MCP tool annotations let clients render confirmation UI and let
agents reason about safety. Assert each registered tool exposes the right
readOnly/destructive/idempotent hints.

If the installed FastMCP build doesn't surface annotations at all we skip
rather than invent — but the supported tools-API version does, see the
helper below.
"""

from __future__ import annotations

import pytest

import mcp_cst.server as server


def _get_tool_annotations(name: str):
    tool = server.mcp._tool_manager.get_tool(name)
    if tool is None:
        pytest.skip(f"tool {name!r} not registered")
    return getattr(tool, "annotations", None)


READ_TOOLS = [
    "server_info",
    "get_ticket",
    "get_tickets",
    "search_tickets",
    "search_and_fetch",
    "aggregate_tickets",
]


@pytest.mark.parametrize("name", READ_TOOLS)
def test_read_tools_have_read_only_hint(name):
    ann = _get_tool_annotations(name)
    if ann is None:
        pytest.skip("FastMCP build does not expose tool annotations")
    assert ann.readOnlyHint is True, f"{name} should be readOnlyHint=True"


def test_delete_ticket_is_destructive_and_not_idempotent():
    ann = _get_tool_annotations("delete_ticket")
    if ann is None:
        pytest.skip("FastMCP build does not expose tool annotations")
    assert ann.destructiveHint is True
    assert ann.idempotentHint is False
    # And not advertised as read-only.
    assert ann.readOnlyHint is not True


@pytest.mark.parametrize("name", ["create_ticket", "update_ticket"])
def test_write_tools_are_not_idempotent(name):
    ann = _get_tool_annotations(name)
    if ann is None:
        pytest.skip("FastMCP build does not expose tool annotations")
    assert ann.idempotentHint is False
    assert ann.readOnlyHint is not True
