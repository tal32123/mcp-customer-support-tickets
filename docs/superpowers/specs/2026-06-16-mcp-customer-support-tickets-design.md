# MCP Customer-Support-Tickets Server — Design Spec

**Date:** 2026-06-16
**Author:** Tal
**Status:** Approved for implementation planning
**Companion:** `design.html` (authoritative visual spec; this doc is the implementation-facing version)

## 1. Purpose

A read-only Model Context Protocol (MCP) server that exposes the public Hugging Face dataset `Tobi-Bueck/customer-support-tickets` (~61,800 rows, EN + DE) to MCP clients (Claude Code, Codex, Claude Desktop). Users ask plain-English questions; the server answers with verbatim dataset content, hybrid retrieval, filterable analytics, and one grounded generation surface (`draft_reply`).

Single core flow: **search → cite → fetch.** Plus an analytics tool and one prompt for drafted replies.

## 2. Scope

**In scope (this spec):**

- Ingest pipeline: HF Parquet → LanceDB with rows + BM25 index + vector embeddings
- 4 tools: `search_tickets`, `get_ticket`, `aggregate_tickets`, `server_info`
- 2 resources: `ticket://{id}`, `schema://tickets`
- 1 prompt: `draft_reply`
- 5 guardrails (G1–G5)
- stdio transport via FastMCP
- Unit-test scaffolding with a ~200-row fixture
- `uv`-driven local-dev and `uvx` publish-and-run workflows

**Out of scope:**

- Remote HTTP transport / OAuth (deferred)
- Cross-encoder reranking on by default (gated behind `RERANK=true` env var)
- Additional prompts (`triage_ticket`, `summarize_trends`)
- PII redaction (deferred — dataset is synthetic-feeling)
- Eval harness, telemetry, observability
- MCP-protocol-level integration tests

## 3. Stack

| Concern | Choice | Why |
|---|---|---|
| Language / runtime | Python 3.13 | Current stable. HF + LanceDB ecosystems are Python-first. |
| Env / build / launcher | `uv` (with `uvx` for clients) | Single tool covers env, deps, packaging, and the launcher MCP clients invoke. |
| MCP SDK | Official Python SDK (FastMCP) | Standard, decorator-based, low ceremony. |
| Store | LanceDB (embedded, columnar) | Rows + BM25 + vectors in one place. Apache 2.0. Scales 100M+. |
| Aggregations | Polars over LanceDB's Arrow output | Zero-copy `group_by` for `aggregate_tickets`. |
| Embeddings | `intfloat/multilingual-e5-small` (384-dim) | Free, local, multilingual (EN + DE). ~120 MB. |
| Drafting LLM | Anthropic SDK (`claude-opus-4-7`) if `ANTHROPIC_API_KEY` set, else OpenAI (`gpt-4o`) if `OPENAI_API_KEY`. | Picked at runtime; refuses with setup hint if neither set. |
| Transport | stdio | Default for Claude Code / Codex / Desktop. No network, no auth. |
| Package layout | src-layout, hatchling backend | Standard modern Python packaging. |
| Cache path | `platformdirs.user_cache_dir("mcp-customer-support-tickets") / <revision> / <model_id> /` | Per-OS conventional. Holds LanceDB store + embedding model. |

## 4. Data model

### 4.1 Dataset facts

- ~61,800 rows, single `train` split
- Languages: English and German only
- Columns: `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `version`, `tag_1`…`tag_6`
- No ticket id column; no timestamps; no customer fields
- License: CC-BY-NC-4.0 (non-commercial)

### 4.2 Derived fields written at ingest

| Field | Definition |
|---|---|
| `id` | `sha1(revision_sha || row_index)[:12]` — stable per-revision. Used everywhere. |
| `tags` | `List<string>`: collapse of `tag_1`…`tag_6` with empties dropped. Used for filtering / aggregation. |
| `tag_1`…`tag_6` | **Preserved verbatim** alongside `tags` (G3 compliance). |
| `embedding` | 384-dim vector of `subject + "\n" + body` from multilingual-e5-small. |
| `text_search` | Concatenation of `subject`, `body`, and `tags` used as the BM25 corpus. |

### 4.3 Revision pinning

The server pins one HF dataset revision at startup. The pin is:

- Surfaced by `server_info`
- Implicitly embedded in every `ticket://{id}` URI because `id = sha1(revision || row_index)[:12]` — same row, different revision → different id. No revision component appears in the URI string itself.
- Part of the cache key (different revision → re-ingest + re-embed)
- Overridable via `MCP_CST_DATASET_REVISION` env var; defaults to a known-good commit hash baked into the package

## 5. Tools

All tools set MCP's `readOnlyHint: true` and `destructiveHint: false`.

### 5.1 `search_tickets`

Hybrid BM25 + vector retrieval. Filter args. Returns previews, not full bodies.

```python
search_tickets(
    q: str,
    queue: str | None = None,
    priority: str | None = None,
    language: Literal["en", "de"] | None = None,
    type: str | None = None,
    tags: list[str] | None = None,       # AND-match against normalized `tags` list
    limit: int = 10,                      # hard cap 50
) -> list[TicketPreview]
```

Each result: `{id, subject, snippet (240 chars of body), score, language, queue, priority}` plus a `ticket://{id}` resource link.

**Retrieval pipeline:**

1. BM25 over `text_search` field, top 50, with `WHERE` filters applied.
2. Embed `q` with multilingual-e5-small.
3. Vector kNN over `embedding`, top 50, same `WHERE` filters.
4. Reciprocal Rank Fusion (RRF, k=60): merge by rank position only — no score normalization, no weights.
5. Return top `limit`.
6. **Deferred:** if `RERANK=true`, run `BAAI/bge-reranker-base` over RRF output before truncation.

### 5.2 `get_ticket`

```python
get_ticket(id: str) -> Ticket
```

Returns every column of the matching row, verbatim, wrapped in `<ticket>` tags per G4. Unknown `id` → structured error (`error.code = "TICKET_NOT_FOUND"`).

### 5.3 `aggregate_tickets`

```python
aggregate_tickets(
    group_by: Literal["queue", "priority", "language", "type", "tags"],
    filters: dict[str, str | list[str]] | None = None,   # same filter shape as search_tickets
) -> list[{group: str, count: int}]
```

Implemented via Polars `group_by` over LanceDB's Arrow output. For `group_by="tags"`, the list field is exploded before counting (a ticket with 3 tags contributes 1 to each).

Unsupported `group_by` value → structured error (`error.code = "UNSUPPORTED_GROUP_BY"`).

### 5.4 `server_info`

```python
server_info() -> {
    "dataset_id": "Tobi-Bueck/customer-support-tickets",
    "dataset_revision": "<sha>",
    "embedding_model": "intfloat/multilingual-e5-small",
    "row_count": 61834,
    "license": "CC-BY-NC-4.0",
    "package_version": "<semver>",
    "rerank_enabled": false,
}
```

No args. Read-only metadata. Useful for "which revision are we on?" diagnostic questions.

## 6. Resources

### 6.1 `ticket://{id}`

Citation handle. The client can attach a specific ticket to the chat by URI. Resolves to the same payload as `get_ticket(id)`, with `<ticket>` wrapping intact.

### 6.2 `schema://tickets`

The canonical schema description: column list, valid values (52 queues, 5 priorities, 2 languages, etc.), and a note that timestamps / customer fields / date ranges are not present.

## 7. Prompt: `draft_reply`

The one generative surface. Mechanics:

```
draft_reply(ticket_id: str, target_language: str | None = None)
```

1. **Resolve target.** Look up the ticket. Missing id → structured error.
2. **Confirm with user.** First prompt message surfaces target's subject + body so the user verifies it's the right ticket before drafting begins.
3. **Retrieve grounding (code, not LLM).**
   - Embed target's `subject + body`.
   - kNN against the store.
   - **Filter:** keep only tickets where `cosine_similarity >= 0.70` AND `answer` is non-empty.
   - **Take up to 5.** Fewer is OK.
4. **Refuse if 0 qualify.** Return structured error `NO_GROUNDING_AVAILABLE` (matches G2 fail-loud). No silent fallback to ungrounded drafting.
5. **Assemble messages.** Prompt hands the LLM:
   - System: "You are drafting a customer-support reply. Follow the style and patterns of the prior answers shown. Reply in `<target_language>`."
   - User content: target ticket wrapped in `<ticket>` tags + 1–5 prior tickets wrapped as `<prior_ticket>…<prior_answer>` pairs.
6. **Inject-refuse the target.** If target's body contains injection-shaped patterns (e.g. "ignore previous instructions"), refuse with `INJECTION_DETECTED` error before invoking the LLM.
7. **Output.** LLM-drafted reply, beginning with: `Based on ticket {target_id}, drawing on {n} prior similar replies ({prior_id_1}, …): …`
8. **Language default.** `target_language` defaults to the target ticket's `language` field.
9. **LLM selection.** Anthropic if `ANTHROPIC_API_KEY` set, else OpenAI, else refuse with `NO_LLM_CONFIGURED`.

**Guardrail update from design.html §11:** the "tone" row flips. The reply IS shaped by the retrieved prior answers — that's the entire point of grounding. The earlier "does not imitate answer style" line is replaced by "follows the style and patterns of the retrieved prior answers."

## 8. Guardrails

| Code | Rule |
|---|---|
| G1 | Bounded responses: default page size 10, hard cap 50. Previews only from search; full bodies only via `get_ticket`. |
| G2 | Fail loud on unsupported queries (date filters, invalid priority, unknown group-by, ungrounded `draft_reply`). Structured error, no silent fallback. |
| G3 | Verbatim only: read endpoints return raw dataset bytes. `draft_reply` is the only place new text is generated. Original `tag_1`…`tag_6` preserved alongside normalized `tags`. |
| G4 | Treat ticket text as untrusted. Wrap returned content in `<ticket>` tags; tool descriptions label content inside the tags as data, not instructions. |
| G5 | Read-only, declared. `readOnlyHint: true` + `destructiveHint: false` on every tool. No write endpoint exists anywhere in the codebase. |

## 9. Error shape

All errors are returned via MCP's standard tool-error mechanism with a JSON payload:

```json
{"error": {"code": "<MACHINE_CODE>", "message": "<human-readable>"}}
```

Defined codes:

- `TICKET_NOT_FOUND`
- `UNSUPPORTED_GROUP_BY`
- `UNSUPPORTED_FILTER` (e.g., date-range)
- `NO_GROUNDING_AVAILABLE` (draft_reply, no ticket clears 70%)
- `INJECTION_DETECTED`
- `NO_LLM_CONFIGURED`
- `DATASET_UNAVAILABLE`

## 10. First-run UX

On first server start:

1. Download embedding model (~120 MB) → cache.
2. Download HF dataset Parquet at pinned revision → cache.
3. Run embedding pass over ~62k rows (~2 min CPU, ~15 s GPU).
4. Build BM25 index in LanceDB.
5. Persist all artifacts under the cache path keyed by `revision + model_id`.

Progress is reported via MCP's `notifications/progress` (1% granularity for the embedding pass). Subsequent server starts use the cache and are sub-second.

Re-ingest is triggered only when `revision` or `model_id` changes.

## 11. Tests

Pytest, unit only. Fixture: ~200 committed rows mirroring the real schema (no HF download in tests).

Coverage targets:

- `id` derivation: stable across runs, unique within fixture
- Schema resource: shape + valid-values list
- `search_tickets`: filter correctness; RRF merge correctness; limit/cap enforcement; preview shape (240-char snippet)
- `get_ticket`: verbatim return, `<ticket>` wrap, not-found error
- `aggregate_tickets`: counts per group correct; tag explosion correct; unsupported group_by errors
- `server_info`: shape
- Guardrail rejections: date filters, invalid priority, unknown group-by, ungrounded draft, injection-shaped input
- `draft_reply`: retrieval filter (70% + non-empty answer); refusal when 0 qualify; injection refusal; LLM selection logic (mock both SDK clients)

No MCP-protocol-level integration tests. FastMCP wiring trusted by inspection.

## 12. Repo layout

```
mcp-customer-support-tickets/
├── pyproject.toml
├── README.md
├── design.html
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-06-16-mcp-customer-support-tickets-design.md
├── src/mcp_cst/
│   ├── __init__.py
│   ├── server.py             # FastMCP entry point, tool/resource/prompt wiring
│   ├── config.py             # env vars, revision pin, cache path, API-key detection
│   ├── data/
│   │   ├── store.py          # LanceDB open/create; row + BM25 + vector access
│   │   ├── ingest.py         # HF Parquet → embedding pass → LanceDB
│   │   └── aggregates.py     # Polars over Arrow for group-by counts
│   ├── retrieval/
│   │   ├── hybrid.py         # BM25 + vector + RRF
│   │   └── rerank.py         # cross-encoder, behind RERANK=true (deferred but stubbed)
│   ├── tools/
│   │   ├── search_tickets.py
│   │   ├── get_ticket.py
│   │   ├── aggregate_tickets.py
│   │   └── server_info.py
│   ├── resources/
│   │   ├── ticket.py
│   │   └── schema.py
│   ├── prompts/
│   │   └── draft_reply.py
│   ├── llm/
│   │   ├── anthropic_client.py
│   │   └── openai_client.py
│   ├── errors.py             # structured error codes + helpers
│   └── safety.py             # injection detector, <ticket> wrap helper
└── tests/
    ├── conftest.py
    ├── fixtures/             # ~200 rows committed
    └── unit/                 # one file per module above
```

## 13. How to run

### Local dev

```sh
git clone <repo-url> mcp-customer-support-tickets
cd mcp-customer-support-tickets
uv sync
uv run mcp-customer-support-tickets   # stdio server
uv run pytest                          # tests
```

### Via PyPI (after publish)

```sh
uvx mcp-customer-support-tickets
```

### Wiring into Claude Code

```json
{
  "mcpServers": {
    "customer-support-tickets": {
      "command": "uvx",
      "args": ["mcp-customer-support-tickets"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MCP_CST_DATASET_REVISION": "main"
      }
    }
  }
}
```

### Environment variables

| Var | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | One of these two for `draft_reply` | Drafting via Claude. Preferred when set. |
| `OPENAI_API_KEY` | ↑ | Drafting via GPT. Fallback. |
| `MCP_CST_DATASET_REVISION` | No | HF dataset revision pin. Defaults to baked-in known-good sha. |
| `MCP_CST_CACHE_DIR` | No | Override cache dir. |
| `RERANK` | No | `true` enables cross-encoder rerank (~270 MB download on first use). |

## 14. Risks (carried forward from design.html §13)

Mitigations remain as in the design HTML; nothing changes from the brainstorming. Key items:

- Hallucinated tickets → G3 verbatim
- Prompt injection via bodies → G4 wrap + injection detector in `draft_reply`
- Dataset drift → revision pin in URI + cache key
- License (CC-BY-NC) → README + `server_info`
- Confident wrong answers → return scores + snippets; G2 fail-loud
- Hebrew quality → documented limitation; `bge-m3` upgrade path noted
- First-run latency → `notifications/progress`; cached after first run
- PII leakage → deferred per direction

## 15. Build sequence (handoff hint for writing-plans)

Order that yields the earliest end-to-end runnable demo:

1. Project skeleton + `pyproject.toml` + `uv sync` working
2. `config.py`, `errors.py`, `safety.py` (no I/O dependencies)
3. `data/ingest.py` + `data/store.py` (LanceDB schema, dataset download, embedding pass)
4. `schema://tickets` resource + `server_info` tool (smallest tools; validate FastMCP wiring)
5. `get_ticket` + `ticket://{id}` (verbatim path)
6. `retrieval/hybrid.py` + `search_tickets` (the headline tool)
7. `aggregate_tickets`
8. `prompts/draft_reply.py` + `llm/` clients
9. Tests in parallel with each module above (TDD-style if writing-plans recommends it)
10. README + publishable package metadata

## 16. Approval

This spec consolidates the design.html and the 2026-06-16 brainstorming session decisions. On user approval, hand off to `superpowers:writing-plans` to produce the implementation plan.
