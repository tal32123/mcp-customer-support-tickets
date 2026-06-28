# mcp-customer-support-tickets

A Model Context Protocol (MCP) server that exposes the
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
dataset (~62k EN/DE tickets) to MCP-capable clients (Claude Code, Codex,
Claude Desktop, Cursor).

Capabilities:

- **search_tickets** — hybrid BM25 + vector retrieval with cursor pagination.
- **search_and_fetch** — search + full-row hydration in one call.
- **aggregate_tickets** — group-by counts (queue / priority / language / type / tags).
- **get_ticket / get_tickets** — verbatim fetch by id.
- **create_ticket / update_ticket / delete_ticket** — CRUD with auto re-embed.
- **draft_reply** — assembles a grounded prompt (target ticket + ≤5 prior tickets with cosine ≥ 0.70 + a type-aware scaffold). No LLM is invoked on the server; the calling client's model writes the reply.

The server **does not call any LLM API** — no `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` to set. The only model that runs locally is
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(384-dim, EN+DE) for retrieval embeddings, loaded once on first start via
`sentence-transformers`. `uv sync` pulls the CUDA-12.4 torch wheel
(~3 GB) — runtime calls `torch.cuda.is_available()` and uses the GPU
when present, transparently falls back to CPU otherwise. Same wheel
works on GPU laptops, CPU laptops, CI, and Railway.

## Required env vars

| Var | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes | Postgres DSN. The target database must have the `pgvector` extension available; the server runs `CREATE EXTENSION IF NOT EXISTS vector` on startup. |
| `MCP_CST_DB_SCHEMA` | no | Schema to create tables in. Defaults to `public`. |
| `MCP_CST_DATASET_REVISION` | no | HF dataset revision pin. Defaults to `main`. |

## How to run

Three paths, pick whichever fits — they all expose the same MCP tools.

### 1. Connect directly to the hosted Railway URL (no local setup)

The server is already deployed and seeded on Railway:

```
https://mcp-customer-support-tickets-production.up.railway.app/mcp
```

Add it to any MCP client that supports remote streamable-http transport.
**Claude Code:**

```sh
claude mcp add --transport http customer-support-tickets \
  https://mcp-customer-support-tickets-production.up.railway.app/mcp
```

**Codex CLI** (`~/.codex/config.toml`):

```toml
[mcp_servers.customer-support-tickets]
url = "https://mcp-customer-support-tickets-production.up.railway.app/mcp"
transport = "http"
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "customer-support-tickets": {
      "url": "https://mcp-customer-support-tickets-production.up.railway.app/mcp",
      "transport": "http"
    }
  }
}
```

### 2. Run locally via Docker Compose

`docker-compose.yml` wires up `pgvector/pgvector:pg17` + the mcp server.
Server listens on `http://localhost:8000/mcp`.

```sh
git clone <repo-url> mcp-customer-support-tickets
cd mcp-customer-support-tickets
docker compose up --build
# first boot: server downloads HF dataset and embeds 62k rows into the
# local pgvector volume. Restarts skip the ingest via the store_meta marker.
```

The compose `mcp` service reserves all NVIDIA GPUs by default (the image
ships the CUDA torch wheel from the pytorch CUDA base image). On a host
with the NVIDIA Container Toolkit installed, embedding runs on the GPU —
which cuts first-boot ingest from ~30 min CPU to a few minutes. On a host
without a GPU / toolkit, comment out the `deploy.resources` block in
`docker-compose.yml` and torch falls back to CPU.

Point any MCP client at `http://localhost:8000/mcp` using the same
`transport = "http"` configs as the Railway URL above (just swap the
domain).

### 3. Run locally via uv (stdio)

```sh
git clone <repo-url> mcp-customer-support-tickets
cd mcp-customer-support-tickets
uv sync

# any pgvector-enabled Postgres works; simplest is a throwaway container:
docker run -d --name mcp-pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17

export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
uv run mcp-customer-support-tickets    # stdio server
uv run pytest                          # tests
```

Wire it to Claude Code:

```sh
claude mcp add customer-support-tickets --scope user \
  -e DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \
  -- uv run --directory /absolute/path/to/mcp-customer-support-tickets mcp-customer-support-tickets
```

Or Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.customer-support-tickets]
command = "uv"
args = ["run", "--directory", "/absolute/path/to/mcp-customer-support-tickets", "mcp-customer-support-tickets"]
env = { DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres" }
```

Or Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "customer-support-tickets": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/mcp-customer-support-tickets", "mcp-customer-support-tickets"],
      "env": { "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/postgres" }
    }
  }
}
```

## RAG eval results

The retrieval surface is evaluated end-to-end against the real
`intfloat/multilingual-e5-small` embedder over the full 62k-row HF
dataset, ingested into a live Postgres + pgvector store, with the
pytest suite querying through the `TicketStore` against that store.
21 tests across 6 dimensions: known-item retrieval, language
behaviour, grounding coherence, semantic quality, latency, robustness.

Run locally:

```bash
TEST_DATABASE_URL=<your-pg-dsn> MCP_CST_EVAL_FULL=1 uv run pytest tests/integration/rag_eval -q -s
```

The eval connects to whatever pgvector instance `TEST_DATABASE_URL`
points at (skip the env var and testcontainers will spin a throwaway
one + ingest 62k rows first, which adds ~30 min on CPU).

### Latest scores

Measured against the live Railway-hosted pgvector (full 62,000-row
corpus, 250 EN + 250 DE seeds, real e5-small embedder, 41 min wall time).

| # | Test | What it measures | Score | Bar | Pass |
|---|---|---|---:|---:|:---:|
| 1 | subject_hit_rate@10 | Real ticket subject as query — is the gold ticket in top 10? | **96.35%** | ≥ 90% | ✓ |
| 2 | subject_MRR@10 | Reciprocal rank of the gold ticket | **0.80** | ≥ 0.75 | ✓ |
| 3 | body_hit_rate@10 | 12-word body slice as query — top 10? | **99.80%** | ≥ 80% | ✓ |
| 4 | body_MRR@10 | Reciprocal rank | **0.91** | ≥ 0.60 | ✓ |
| 5 | body_NDCG@10 | Rank-weighted relevance | **0.93** | ≥ 0.65 | ✓ |
| 6 | per-lang body@10 EN | Body-slice known-item, EN subset | **100.00%** | ≥ 75% | ✓ |
| 7 | per-lang body@10 DE | Body-slice known-item, DE subset | **99.60%** | ≥ 75% | ✓ |
| 9 | lang_purity DE | 30 free-text DE queries with no filter; mean % of top-10 in German | **100%** | ≥ 90% | ✓ |
| 10 | lang_purity EN | 30 free-text EN queries with no filter; mean % of top-10 in English | **73.33%** | ≥ 70% | ✓ |
| 11 | language_filter_pushdown | When `language=` is set, every hit is that language | **pass** | hard | ✓ |
| 13 | pagination_integrity | Walk every page — zero duplicates, total matches estimate | **pass** | hard | ✓ |
| 14 | grounding_type_coherence | `draft_reply` picks grounding sharing target's type | **80.67%** | ≥ 60% | ✓ |
| 15 | grounding_lang_coherence | `draft_reply` picks grounding in target's language | **100%** | ≥ 95% | ✓ |
| 16 | topical_intent_top3 | 20 hand-written natural queries; relevant hit in top-3 | **75%** (15/20) | ≥ 80% | ✗ |
| 17 | topical_intent_top10 | Same, top-10 | **80%** (16/20) | ≥ 85% | ✗ |
| 18 | cross_lingual_recall (diagnostic) | DE query → EN match (and vice versa); recorded only, not asserted | **1/6** | — | — |
| 19 | hard_negative_contamination | Off-topic hits in top-3 across 5 topic pairs | **0** | ≤ 2 | ✓ |
| 20 | latency_p50 | Median end-to-end search time (incl. network) | **648 ms** | ≤ 400 ms | ✗ |
| 21 | latency_p95 | 95th percentile | **675 ms** | ≤ 1000 ms | ✓ |
| 22 | robustness_clean | Hit-rate@10 on 30 unmodified seed queries | **100%** | — | ✓ |
| 23 | robustness_lowercase | Same queries, all-lowercase | **100%** | drop ≤ 0.15 | ✓ |
| 24 | robustness_swap_chars | Same queries, with one adjacent-character swap | **70%** | drop ≤ 0.15 | ✗ |
| 25 | robustness_drop_word | Same queries, with one word removed | **100%** | drop ≤ 0.15 | ✓ |

Two HE-language cases (#8 and #12) skip — the live HF dataset ships
only EN and DE rows.

**4 failures, explained:**
- **#20 latency p50 (648 ms vs ≤400 ms bar)**: bar was set for a local
  sub-second store; the live eval measures over MCP + HTTPS + an
  intercontinental hop to the EU. Pure infrastructure cost, not a
  retrieval regression.
- **#24 robustness char-swap (70%)**: with the full 62k corpus and
  Postgres FTS tokenization, a typo'd query genuinely loses BM25
  signal. Smaller corpus could mask this.
- **#16 / #17 topical_intent (75% / 80%)**: at 62k rows there are many
  more topical competitors than the 2k-sample baseline, so queries like
  "I want a refund" hit billing/exchange neighbours instead of the
  refund ticket. Honest ranking-at-scale ceiling for e5-small.

Known limitation: cross-lingual recall (#18) stays low on this
embedder + dataset combination — kept as a diagnostic so improvements
(bigger embedder, query expansion, a re-ranker) surface as a score
change rather than a silent regression.

## License

Server code: MIT. Dataset (CC-BY-NC-4.0) is non-commercial — see
`server_info` for the license string surfaced at runtime.
