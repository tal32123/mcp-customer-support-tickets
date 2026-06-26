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

The image bakes CPU-only PyTorch **and the embedding model weights** so the
server is ready to ingest the moment it starts. Mount a volume at `/data` so
the LanceDB store survives restarts — otherwise every cold start re-embeds the
~62k-row corpus from scratch.

```sh
docker build -t mcp-customer-support-tickets .
docker run --rm -p 8000:8000 -v mcp_data:/data mcp-customer-support-tickets
# server now listening on http://localhost:8000/mcp
```

Or with compose (named volume `mcp_data` persists between `up`/`down`):

```sh
docker compose up --build
```

First boot on an empty volume: ~60-90 s for the 62k-row embed pass (model is
already baked in). After that the volume is warm and cold starts are seconds.

## Deploying to Railway

1. `railway init` in this repo, then `railway up` — `railway.json` points the
   builder at the `Dockerfile`.
2. In the service settings, attach a **Volume** mounted at `/data` (1 GB is
   enough; 2 GB gives headroom for HF cache growth).
3. No env vars are required — the Dockerfile already sets
   `MCP_TRANSPORT=streamable-http` and `MCP_CST_CACHE_DIR=/data`. Railway
   injects `PORT` automatically. The HF cache lives inside the image
   (`/opt/hf-cache`), not the volume.
4. Point a remote MCP client at `https://<your-app>.up.railway.app/mcp`.

## First-run notes

The first time the server starts, it downloads the embedding model
(~120 MB) and the HF Parquet, then runs an embedding pass over ~62k
tickets (~2 min on CPU). Everything is cached on disk, keyed by
dataset revision and model id. Subsequent starts are sub-second.

New tickets created via `create_ticket` (and edits via `update_ticket` / `delete_ticket`) are written into the same per-revision cache directory and persist across restarts. Bumping `MCP_CST_DATASET_REVISION` invalidates the cache and triggers a fresh ingest — any locally-created tickets in the old cache are not migrated.

## License

Server code: MIT. Dataset (CC-BY-NC-4.0) is non-commercial — see
`server_info` for the license string surfaced at runtime.
