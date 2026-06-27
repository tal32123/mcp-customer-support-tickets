# mcp-customer-support-tickets

A Model Context Protocol (MCP) server that exposes the
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
dataset to MCP-capable clients (Claude Code, Claude Desktop, Codex, Cursor).

- **Search** by hybrid BM25 + vector across ~62k EN/DE support tickets.
- **Fetch** any ticket verbatim by id.
- **Aggregate** counts by queue, priority, language, type, or tags.
- **Assemble** a grounded draft-reply prompt: target ticket + up to 5 prior tickets+answers (cosine >= 0.70) + a type-aware scaffold the caller's LLM fills in. No API key needed.
- **Create** new tickets locally (subject + body + optional metadata) and get back a stable 12-char id; they're immediately searchable. No LLM is invoked on the server — pass already-composed text.
- **Update** an existing ticket's fields by id; re-embeds and re-indexes so changes are immediately searchable.
- **Delete** a ticket by id. Destructive and irreversible within the running store.

## Models

This server **does not call any LLM API** — there is no `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` to set. Generation (e.g. drafting a reply) is performed by the
calling MCP client's own model (Claude in Claude Code / Desktop, GPT in Codex,
etc.); the server only assembles grounded context and a scaffold for it.

The one model that runs locally is for **retrieval embeddings**:

| Purpose | Model | Where it runs | How to change it |
|---|---|---|---|
| Query + ticket embeddings (384-dim, multilingual EN+DE) | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | On-device via `sentence-transformers` (CPU or CUDA if available) | Edit `EMBEDDING_MODEL` in `src/mcp_cst/config.py`. Changing it invalidates the on-disk embedding cache and triggers a fresh ~62k-row embedding pass on next start. |

First run downloads ~120 MB of model weights to the HuggingFace cache; all
subsequent starts are offline and sub-second.

## Install

Clone the repo somewhere stable and reference its absolute path in the commands below.

### Claude Code (CLI, no JSON editing)

```sh
claude mcp add customer-support-tickets --scope user \
  -- uv run --directory /path/to/mcp-customer-support-tickets mcp-customer-support-tickets
```

Drop `--scope user` for cwd-only. Verify with `claude mcp list`.

### Codex CLI

Add this block to `~/.codex/config.toml`:

```toml
[mcp_servers.customer-support-tickets]
command = "uv"
args = ["run", "--directory", "/path/to/mcp-customer-support-tickets", "mcp-customer-support-tickets"]
```

### Claude Desktop (and any other JSON-config client)

Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "customer-support-tickets": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-customer-support-tickets", "mcp-customer-support-tickets"]
    }
  }
}
```

## Local development

```sh
git clone <repo-url> mcp-customer-support-tickets
cd mcp-customer-support-tickets
uv sync
uv run mcp-customer-support-tickets   # stdio server
uv run pytest                          # tests
```

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `MCP_CST_DATASET_REVISION` | no | HF dataset revision pin. Defaults to `main`. |
| `MCP_CST_CACHE_DIR` | no | Override cache dir. Defaults to `platformdirs.user_cache_dir("mcp-customer-support-tickets")`. |
| `MCP_TRANSPORT` | no | `stdio` (default) or `streamable-http` for remote hosting. |
| `MCP_HOST` | no | Bind host when `MCP_TRANSPORT=streamable-http`. Defaults to `0.0.0.0`. |
| `PORT` / `MCP_PORT` | no | Bind port when `MCP_TRANSPORT=streamable-http`. Defaults to `8000`. `PORT` wins (Railway/Fly convention). |

## Docker

The image is built on the official PyTorch CUDA base and bakes the
embedding model weights and **a fully-ingested LanceDB store** at
`/opt/store-seed`. On first boot the entrypoint copies the seed into
`/data` if the volume is empty, so the server is ready in seconds — the
62k-row embed pass already ran at build time. Mount a volume at `/data`
so writes (new tickets) survive restarts.

```sh
docker build -t mcp-customer-support-tickets .
docker run --rm -p 8000:8000 -v mcp_data:/data mcp-customer-support-tickets
# server now listening on http://localhost:8000/mcp
```

Or with compose (named volume `mcp_data` persists between `up`/`down`):

```sh
docker compose up --build
```

On hosts without a GPU (Railway, CI, CPU laptops) torch initialises
lazily and runs CPU-only — no errors, just slower. On an NVIDIA host with
the NVIDIA Container Toolkit installed, pass `--gpus all` to actually use
the GPU:

```sh
docker run --rm --gpus all -p 8000:8000 -v mcp_data:/data mcp-customer-support-tickets
```

Build time depends on the host: ~3-5 min on Linux/CI, ~3 min on a GPU
builder, ~30 min on Docker Desktop on Windows without GPU passthrough
(the embed pass over 62k rows runs once at build). BuildKit caches the
ingest layer, so subsequent rebuilds reuse it unless ingest code or the
dataset revision changes. Runtime first boot on an empty volume is a few
seconds (seed copy + LanceDB open). Final image is ~5-6 GB (CUDA torch +
cuDNN ship as part of the base).

## Deploying to Railway

The image is built once per push to `main` by GitHub Actions
(`.github/workflows/ci.yml`) and pushed to
`ghcr.io/tal32123/mcp-customer-support-tickets:latest`. Railway pulls the
pre-built image (`railway.json` → `build.image`) instead of building from
source, so deploys are a ~30 s pull and don't depend on Railway's
builder.

1. `railway init` in this repo and link the service to GitHub so a push
   to `main` triggers a redeploy after the GHA build finishes. Or trigger
   redeploys manually after the image tag updates.
2. The ghcr package must be **public** for Railway to pull anonymously on
   the Hobby plan (Repo → Packages → set visibility → Public). Private
   pulls require Railway Pro.
3. In the service settings, attach a **Volume** mounted at `/data` (1 GB
   is enough; 2 GB gives headroom for HF cache growth).
4. No env vars are required — the image already sets
   `MCP_TRANSPORT=streamable-http` and `MCP_CST_CACHE_DIR=/data`. Railway
   injects `PORT` automatically. The HF cache lives inside the image
   (`/opt/hf-cache`), not the volume.
5. Point a remote MCP client at `https://<your-app>.up.railway.app/mcp`.

## First-run notes

The first time the server starts, it downloads the embedding model
(~120 MB) and the HF Parquet, then runs an embedding pass over ~62k
tickets (~2 min on CPU). Everything is cached on disk, keyed by
dataset revision and model id. Subsequent starts are sub-second.

New tickets created via `create_ticket` (and edits via `update_ticket` / `delete_ticket`) are written into the same per-revision cache directory and persist across restarts. Bumping `MCP_CST_DATASET_REVISION` invalidates the cache and triggers a fresh ingest — any locally-created tickets in the old cache are not migrated.

## RAG eval results

The retrieval surface is evaluated end-to-end against the real
`intfloat/multilingual-e5-small` embedder over a 2,000-row stratified
sample (1k EN + 1k DE) of the live HF dataset. 21 tests across 6
dimensions: known-item retrieval, language behaviour, grounding
coherence, semantic quality, latency, robustness.

Run locally:

```bash
MCP_CST_EVAL_FULL=1 uv run pytest tests/integration/rag_eval -q -s
```

First run downloads the ~470 MB model and HF dataset (~5–10 min on
CPU); subsequent runs hit the local cache (~60 s).

### Latest scores

| # | Test | What it measures | Score | Bar |
|---|---|---|---:|---:|
| 1 | Find ticket by subject (top-10) | Take a real ticket, search using its subject — does it come back in top 10? | **97.4%** | ≥ 90% |
| 2 | Find ticket by subject (rank quality) | Reciprocal rank of the gold ticket — closer to 1.0 means usually #1 | **0.84** | ≥ 0.75 |
| 3 | Find ticket by body slice (top-10) | Search using a 12-word slice of the body — does the ticket come back? | **97.6%** | ≥ 80% |
| 4 | Find ticket by body slice (rank quality) | Reciprocal rank of the gold ticket | **0.87** | ≥ 0.60 |
| 5 | Find ticket by body (NDCG@10) | Rank-weighted relevance score | **0.90** | ≥ 0.65 |
| 6 | Per-language retrieval — EN | Same as #3, English subset only | **98.4%** | ≥ 75% |
| 7 | Per-language retrieval — DE | Same as #3, German subset only | **96.8%** | ≥ 75% |
| 9 | Language purity — DE | 30 free-text DE queries with no filter; mean % of top-10 in German | **99.0%** | ≥ 90% |
| 10 | Language purity — EN | 30 free-text EN queries with no filter; mean % of top-10 in English | **80.7%** | ≥ 70% |
| 11 | Language filter pushdown | When `language=` is set, are 100% of results that language? | **100%** | hard pass |
| 13 | Pagination integrity | Walk every page — zero duplicates, total matches estimate | **pass** | hard pass |
| 14 | Grounding coherence — type | `draft_reply` picks grounding tickets sharing the target's type | **60.9%** | ≥ 60% |
| 15 | Grounding coherence — language | `draft_reply` picks grounding tickets in the target's language | **100%** | ≥ 95% |
| 16 | Topical intent — top-3 | 20 hand-written natural queries; relevant hit in top-3 | **90%** (18/20) | ≥ 80% |
| 17 | Topical intent — top-10 | Same, top-10 | **90%** (18/20) | ≥ 85% |
| 18 | Cross-lingual recall (diagnostic) | DE query → EN match (and vice versa) — recorded only, not asserted | **0/6** | — |
| 19 | Hard negatives | Count of pure-off-topic hits in top-3 across 5 topic pairs | **0** | ≤ 2 |
| 20 | Latency — p50 | Median end-to-end search time | **107 ms** | ≤ 400 ms |
| 21 | Latency — p95 | 95th percentile end-to-end search time | **156 ms** | ≤ 1000 ms |
| 22 | Robustness — clean | Hit-rate@10 on 30 unmodified seed queries (baseline for #23–25) | **100%** | — |
| 23 | Robustness — lowercase | Same queries, all-lowercase | **100%** | drop ≤ 0.15 |
| 24 | Robustness — char-swap typo | Same queries, with one adjacent-character swap | **100%** | drop ≤ 0.15 |
| 25 | Robustness — drop a word | Same queries, with one word removed | **100%** | drop ≤ 0.15 |

Two HE-language parametrize cases (#8 and #12) skip — the live HF
dataset ships only EN and DE rows.

**Known limitation**: cross-lingual recall (#18) is 0/6 on this
embedder + dataset combination. The test is intentionally kept as a
diagnostic so improvements (bigger embedder, query expansion, a
re-ranker) show up as a score change rather than a silent regression.

## License

Server code: MIT. Dataset (CC-BY-NC-4.0) is non-commercial — see
`server_info` for the license string surfaced at runtime.
