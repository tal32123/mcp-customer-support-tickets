"""Helpers for treating ticket text as untrusted data."""

from __future__ import annotations
import re
import unicodedata
from xml.sax.saxutils import escape, quoteattr


# The dotall flag lets `.` match newlines so multi-line variants like
# "ignore\nprevious instructions" are still caught.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE | re.DOTALL),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE | re.DOTALL),
    re.compile(r"system\s+prompt\s*[:=]", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\s*[:=]", re.IGNORECASE),
]


_EXTRA_ENTITIES = {'"': "&quot;", "'": "&apos;"}


def escape_text(text: str | None) -> str:
    """XML-escape a string for embedding inside a tag body.

    Use whenever you splice untrusted ticket content into an LLM prompt and
    aren't routing it through `wrap_ticket`. Escapes `& < >` plus `' "` so
    nothing in the content can close a surrounding tag or attribute.

    Accepts `None` and renders it as the empty string — historical stores
    built before the ingest fix can contain null cells, and the prompt
    paths must not crash on them.
    """
    return escape(text or "", _EXTRA_ENTITIES)


def _normalize(text: str) -> str:
    """NFKC-normalize then replace zero-width and other format-category code
    points with a space. Folds homoglyphs and invisible characters that an
    attacker might use to slip a pattern past plain-text regex matching.

    Not a security boundary -- see `looks_like_injection`.
    """
    nfkc = unicodedata.normalize("NFKC", text)
    return "".join(" " if unicodedata.category(ch) == "Cf" else ch for ch in nfkc)


def looks_like_injection(text: str) -> bool:
    """True if `text` contains language commonly used in prompt-injection
    attacks.

    HEURISTIC, NOT A SECURITY GUARANTEE. The patterns are English-only and
    even with NFKC + format-stripping a determined attacker can paraphrase
    around them. Use this as one signal alongside the `<ticket>`-tag
    convention and the LLM-side reminder; do not rely on it as a sole
    defense against malicious ticket content.
    """
    normalized = _normalize(text)
    return any(p.search(normalized) for p in _INJECTION_PATTERNS)


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
