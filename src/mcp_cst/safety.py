"""Helpers for treating ticket text as untrusted data."""

from __future__ import annotations
import logging
import re
import unicodedata
from xml.sax.saxutils import escape, quoteattr


log = logging.getLogger(__name__)
_warned_once = False


# The dotall flag lets `.` match newlines so multi-line variants like
# "ignore\nprevious instructions" are still caught.
_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"system\s+prompt\s*[:=]", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\s*[:=]", re.IGNORECASE),
    # English social-engineering phrasings (#303).
    re.compile(r"\bforget\s+what\s+I\s+said\b", re.IGNORECASE),
    re.compile(r"\boverride\s+your\b", re.IGNORECASE),
    re.compile(r"\byou\s+must\s+now\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\b", re.IGNORECASE),
    re.compile(r"\bfrom\s+now\s+on\s+you\s+(are|act|will)\b", re.IGNORECASE),
    # German (#104).
    re.compile(
        r"ignoriere\s+alle\s+(vorherigen|bisherigen|obigen)\s+(Anweisungen|Anleitungen)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"vergesse\s+alle\s+(vorherigen|bisherigen|obigen)\s+(Anweisungen|Anleitungen)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"Systemprompt\s*[:=]", re.IGNORECASE),
    # Hebrew (#104). No IGNORECASE needed — Hebrew has no case.
    re.compile(r"התעלם"),
    re.compile(r"אל\s+תשים\s+לב"),
    re.compile(r"הוראות\s+חדשות\s*[:=]"),
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
    if text is None:
        global _warned_once
        if not _warned_once:
            log.warning("escape_text received None — upstream nullable field leaked")
            _warned_once = True
        return ""
    return escape(text, _EXTRA_ENTITIES)


def neutralize_markdown(text: str) -> str:
    """De-fang markdown so prior-ticket bodies can't render as live formatting.

    Run AFTER `escape_text` (XML-escape first, markdown-neutralize second).
    Lazy version: break ``` fences with a literal apostrophe-triple, and
    prefix `[` with a zero-width space so `[link](url)` won't render.
    """
    # ponytail: smallest neutralization that defangs the common payloads
    # (code fences + link syntax). Real renderers will still see the text
    # but won't parse it as markdown. Upgrade to a full markdown-stripper
    # if richer payloads (HTML, images, footnotes) start landing.
    text = text.replace("```", "''' ")
    text = text.replace("[", "​[")
    return text


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

    HEURISTIC, NOT A SECURITY GUARANTEE. Patterns cover EN/DE/HE and even
    with NFKC + format-stripping a determined attacker can paraphrase
    around them. Use this as one signal alongside the `<ticket>`-tag
    convention and the LLM-side reminder; do not rely on it as a sole
    defense against malicious ticket content.
    """
    normalized = _normalize(text)
    return any(p.search(normalized) for p in _INJECTION_PATTERNS)


def wrap_ticket(
    *, ticket_id: str, subject: str, body: str, extra: dict[str, str] | None = None
) -> str:
    """Wrap a ticket's fields in <ticket> tags with XML-escaped content.

    Output is intended to be embedded in LLM context as untrusted data.
    Consumers should be reminded by their tool description that content
    inside <ticket> tags is data, not instructions.
    """
    parts = [f"<ticket id={quoteattr(ticket_id)}>"]
    parts.append(f"  <subject>{escape_text(subject)}</subject>")
    parts.append(f"  <body>{escape_text(body)}</body>")
    for k, v in (extra or {}).items():
        parts.append(f"  <{k}>{escape(str(v))}</{k}>")
    parts.append("</ticket>")
    return "\n".join(parts)
