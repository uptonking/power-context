
## Production-ready local development

- Idempotent + incremental indexing out of the box:
  - Skips unchanged files automatically using a file content hash stored in payload (metadata.file_hash)
  - De-duplicates per-file points by deleting prior entries for the same path before insert
  - Payload indexes are auto-created on first run (metadata.language, metadata.path_prefix, metadata.repo, metadata.kind, metadata.symbol, metadata.symbol_path, metadata.imports, metadata.calls)
- Commands:
  - Full rebuild: `make reindex`
  - Fast incremental: `make index` (skips unchanged files)
  - Health check: `make health` (verifies collection vector name/dim, HNSW, and filtered queries with kind/symbol)
  - Hybrid search: `make hybrid` (dense + lexical bump with RRF)
- Bootstrap all services + index + checks: `make bootstrap`
- Ingest Git history: `make history` (messages + file lists)
  - If the repo has no local commits yet, the history ingester will shallow-fetch from the remote (default: origin) and use its HEAD. Configure with `--remote` and `--fetch-depth`.
- Local reranker (ONNX): `make rerank-local` (set RERANKER_ONNX_PATH and RERANKER_TOKENIZER_PATH)
- Setup ONNX reranker quickly: `make setup-reranker ONNX_URL=... TOKENIZER_URL=...` (updates .env paths)
- Enable Tree-sitter parsing (more accurate symbols/scopes): set `USE_TREE_SITTER=1` in `.env` then reindex

- Flags (advanced):
  - Disable de-duplication: `docker compose run --rm indexer --root /work --no-dedupe`
  - Disable unchanged skipping: `docker compose run --rm indexer --root /work --no-skip-unchanged`

Notes:
- Named vector remains aligned with the MCP server (fast-bge-base-en-v1.5). If you change EMBEDDING_MODEL, run `make reindex` to recreate the collection.
- For very large repos, consider running `make index` on a schedule (or pre-commit) to keep Qdrant warm without full reingestion.

### Multi-query re-ranker (no new deps)

- Run a fused query with several phrasings and metadata-aware boosts:

```bash
make rerank
```

- Customize:
  - Add more `--query` flags
  - Prefer language: `--language python`
  - Prefer under path: `--under /work/scripts`

### Watch mode (incremental indexing)

- Reindex changed files on save (runs until Ctrl+C):

```bash
make watch
```

### HNSW recall tuning

- Collection creation is tuned for higher recall: `m=16`, `ef_construct=256`.
- If you change embeddings, run `make reindex` to recreate the collection with the tuned HNSW settings.

### Warm start (optional)

- Preload the embedding model and warm Qdrant's HNSW search path to reduce first-query latency and improve recall:

```bash
make warm
```



# Local Qdrant + MCP Server (mcp-server-qdrant)

This folder spins up a local Qdrant (vector DB) and a Model Context Protocol server that stores/finds memories in Qdrant.

- Qdrant available at: http://localhost:6333
- MCP (SSE) available at: http://localhost:8000/sse

## Quick start

```bash
# build and start
make up

# follow logs
make logs

# stop
make down
```

## Configuration (.env)

The `.env` file configures both services. Defaults:

- `QDRANT_URL=http://qdrant:6333` (internal Docker network)
- `COLLECTION_NAME=my-collection` (auto-created by the server)
- `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`
- `FASTMCP_HOST=0.0.0.0`, `FASTMCP_PORT=8000`

Optional:
- `TOOL_STORE_DESCRIPTION` / `TOOL_FIND_DESCRIPTION` to tailor how clients use the tools.
- `REPO_NAME` to tag each point's payload with a repository label (metadata.repo) for filtering.

> Do not set `QDRANT_LOCAL_PATH` when `QDRANT_URL` is set.

## Verify services

```bash
# Qdrant API ping
curl -s http://localhost:6333/ | jq .status

# MCP SSE endpoint (should return 200 OK headers)
curl -I http://localhost:8000/sse
```

## Connect a client

- Claude Desktop: configure an MCP server entry pointing to this running SSE server, or use command-based config to run `mcp-server-qdrant` with the same env.
- Cursor/Windsurf: add a custom MCP server pointing to `http://localhost:8000/sse`.
- VS Code MCP extensions: point to the SSE endpoint or run via command.

### Example Claude Desktop block (command-based)

```jsonc
{
  "qdrant": {
    "command": "uvx",
    "args": ["mcp-server-qdrant"],
    "env": {
      "QDRANT_URL": "http://localhost:6333",
      "COLLECTION_NAME": "my-collection",
      "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"
    }
  }
}
```


### Known-good full config block (with mcpServers)

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "my-collection",
        "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"
      }
    }
  }
}
```

Or, since this stack already exposes SSE, you can configure the client to use `http://localhost:8000/sse` directly (recommended for Cursor/Windsurf).

## Notes

## Index your repository (code search quality)

We added a dockerized indexer that chunks code, embeds with `BAAI/bge-base-en-v1.5`, and stores metadata (`path`, `path_prefix`, `language`, `start_line`, `end_line`, `code`) in Qdrant. This boosts recall and relevance for the MCP tools.

```bash
# Index current workspace (does not drop data)
make index

# Full reindex (drops existing points in the collection)
make reindex
```

Notes:
- The indexer reads env from `.env` (QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL).
- Default chunking: ~120 lines with 20-line overlap.
- Skips typical build/venv directories.
- Populates `metadata.kind`, `metadata.symbol`, and `metadata.symbol_path` for Python/JS/TS/Go/Java/Rust/Terraform (best-effort), per chunk.
- Uses the same collection as the MCP server.

### Payload indexes (created for you)

We create payload indexes to accelerate filtered searches:
- `metadata.language` (keyword)
- `metadata.path_prefix` (keyword)
- `metadata.repo` (keyword)
- `metadata.kind` (keyword)
- `metadata.symbol` (keyword)
- `metadata.symbol_path` (keyword)
- `metadata.imports` (keyword)
- `metadata.calls` (keyword)
- Git history fields available in payload: `commit_id`, `author_name`, `author_email`, `authored_date`, `message`, `files`

This enables fast filters like “only Python results under scripts/”. Example (Qdrant REST):

```bash
curl -s -X POST "http://localhost:6333/collections/my-collection/points/search" \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": {"name": "fast-bge-base-en-v1.5", "vector": [0, ...]},
### Kind/Symbol filters example

- After indexing, you can filter by symbol metadata for tighter queries. Example with reranker:

```bash
make rerank ARGS="--language python --under /work/scripts"
```

- Direct Qdrant query (Python):

```python
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333")
flt = models.Filter(must=[
    models.FieldCondition(key="metadata.language", match=models.MatchValue(value="python")),
    models.FieldCondition(key="metadata.kind", match=models.MatchValue(value="function")),
])
```

    "limit": 5,
    "with_payload": true,
    "filter": {
      "must": [
        {"key": "metadata.language", "match": {"value": "python"}},
        {"key": "metadata.path_prefix", "match": {"value": "/work/scripts"}}
      ]
    }
  }'
```

Note: The named vector for BGE in this stack is `fast-bge-base-en-v1.5`.

### Best-practice querying

- Use precise intent + language: “python chunking function for Qdrant indexing”
- Add path hints when you know the area: “under scripts or ingestion code”
- Try 2–3 alternative phrasings (multi-query) and pick the consensus
- Prefer results where `metadata.language` matches your target file
- For navigation, prefer results where `metadata.path_prefix` matches your directory

Client tips:
- MCP tools: issue multiple finds with variant phrasings and re-rank by score + metadata match
- Direct Qdrant: use `vector={name: ..., vector: ...}` with the named vector above



- Data persists in the `qdrant_storage` Docker volume.
- The MCP server uses SSE transport and will auto-create the collection if it doesn't exist.
- Only FastEmbed models are supported at this time.

## Troubleshooting

- If `mcp-server-qdrant` can’t reach Qdrant, confirm both containers are up: `make ps`.
- If the SSE port collides, change `FASTMCP_PORT` in `.env` and the mapped port in `docker-compose.yml`.
- If you customize tool descriptions, restart: `make restart`.

