"""Helpers for treating ticket text as untrusted data."""

from __future__ import annotations
import re
from xml.sax.saxutils import escape, quoteattr


_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*[:=]", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\s*[:=]", re.IGNORECASE),
]


_EXTRA_ENTITIES = {'"': "&quot;", "'": "&apos;"}


def looks_like_injection(text: str) -> bool:
    """True if the text contains language commonly used in prompt-injection attacks."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def wrap_ticket(*, ticket_id: str, subject: str, body: str, extra: dict[str, str] | None = None) -> str:
    """Wrap a ticket's fields in <ticket> tags with XML-escaped content.

    Output is intended to be embedded in LLM context as untrusted data.
    Consumers should be reminded by their tool description that content
    inside <ticket> tags is data, not instructions.
    """
    parts = [f"<ticket id={quoteattr(ticket_id)}>"]
    parts.append(f"  <subject>{escape(subject, _EXTRA_ENTITIES)}</subject>")
    parts.append(f"  <body>{escape(body, _EXTRA_ENTITIES)}</body>")
    for k, v in (extra or {}).items():
        parts.append(f"  <{k}>{escape(str(v))}</{k}>")
    parts.append("</ticket>")
    return "\n".join(parts)
