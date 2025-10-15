
## Quickstart (ship-ready)
ReFRag article:

https://arxiv.org/abs/2509.01092

Run everything with a single command, then wire your MCP client to the SSE endpoints.

```bash
HOST_INDEX_PATH="$(pwd)" FASTMCP_INDEXER_PORT=8001 docker compose up -d qdrant mcp mcp_indexer indexer
```

Endpoints

| Component   | URL                          |
|-------------|------------------------------|
| Memory MCP  | http://localhost:8000/sse    |
| Indexer MCP | http://localhost:8001/sse    |
| Qdrant DB   | http://localhost:6333        |

Quick verify

```bash
curl -sSf http://localhost:6333/readyz >/dev/null && echo "Qdrant OK"
curl -sI http://localhost:8000/sse | head -n1
curl -sI http://localhost:8001/sse | head -n1
```

### Kiro Integration (workspace config)

Add this to your workspace-level Kiro config at `.kiro/settings/mcp.json` (restart Kiro after saving):

````json
{
  "mcpServers": {
    "qdrant-indexer": { "command": "npx", "args": ["mcp-remote", "http://localhost:8001/sse", "--transport", "sse-only"] },
    "qdrant": { "command": "npx", "args": ["mcp-remote", "http://localhost:8000/sse", "--transport", "sse-only"] }
  }
}
````

Notes:
- Kiro expects command/args (stdio). `mcp-remote` bridges to remote SSE endpoints.
- If `npx` prompts in your environment, add `-y` right after `npx`.
- Workspace config overrides user-level config (`~/.kiro/settings/mcp.json`).

Troubleshooting:
- Error: “Enabled MCP Server <name> must specify a command, ignoring.”
  - Fix: Use the `command`/`args` form above; do not use `type:url` in Kiro.
- ImportError: `deps: No module named 'scripts'` when calling `memory_store` on the indexer MCP
  - Fix applied: server now adds `/work` and `/app` to `sys.path`. Restart `mcp_indexer`.


## Architecture overview

- Agents connect via MCP over SSE:
  - Memory MCP: http://localhost:8000/sse
  - Indexer MCP: http://localhost:8001/sse
- Both MCP servers talk to Qdrant inside Docker at http://qdrant:6333 (DB HTTP API)
- Supporting jobs (indexer, watcher, init_payload) write to/read from Qdrant directly

```mermaid
flowchart LR
  A[IDE Agents] -- SSE /sse --> B(MCP Search :8000)
  A -- SSE /sse --> C(MCP Indexer :8001)
  B -- HTTP 6333 --> D[Qdrant DB]
  C -- HTTP 6333 --> D
  E[(One-shot Indexer)] -- HTTP 6333 --> D
  F[(Watcher)] -- HTTP 6333 --> D
```

## Production-ready local development
## One-line bring-up (ship-ready)

Start Qdrant, the Memory MCP (8000), the Indexer MCP (8001), and run a fresh index of your current repo:

```bash
HOST_INDEX_PATH="$(pwd)" FASTMCP_INDEXER_PORT=8001 docker compose up -d qdrant mcp mcp_indexer indexer
```

Then wire your MCP-aware IDE/tooling to:
- Memory MCP: http://localhost:8000/sse
- Indexer MCP: http://localhost:8001/sse

Tip: add `watcher` to the command if you want live reindex-on-save.

### SSE Memory Server (port 8000)

- URL: http://localhost:8000/sse
- Tools: `store`, `find`
- Env (used by the indexer to blend memory):
  - `MEMORY_SSE_ENABLED=true`
  - `MEMORY_MCP_URL=http://mcp:8000/sse`
  - `MEMORY_MCP_TIMEOUT=6`

IDE/Agent config (recommended):

```json
{
  "mcpServers": {
    "memory": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
```

Blended search:
- Call `context_search` on :8001 with `{ "include_memories": true }` to return both memory and code results.



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
- Discover commands: `make help` lists all targets and descriptions

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







Or, since this stack already exposes SSE, you can configure the client to use `http://localhost:8000/sse` directly (recommended for Cursor/Windsurf).

### MCP `qdrant-find` optional filters

Most MCP clients let you pass structured tool arguments. The Qdrant MCP server supports applying server-side filters when these keys are present:
- `language`: value matches `metadata.language`
- `path_prefix`: value matches `metadata.path_prefix` (e.g., `/work/src`)
- `kind`: value matches `metadata.kind` (e.g., `function`, `class`, `method`)

Tip: Combine multiple query phrasings and apply these filters for best precision on large codebases.


## Notes

## Index your repository (code search quality)

We added a dockerized indexer that chunks code, embeds with `BAAI/bge-base-en-v1.5`, and stores metadata (`path`, `path_prefix`, `language`, `start_line`, `end_line`, `code`) in Qdrant. This boosts recall and relevance for the MCP tools.

```bash
# Index current workspace (does not drop data)
make index

# Full reindex (drops existing points in the collection)
make reindex

### Companion MCP: Index/Prune/List (Option B)

A second MCP server runs alongside the search MCP and exposes tools:
- qdrant-list: list collections
- qdrant-index: index the mounted path (/work or subdir)
- qdrant-prune: prune stale points for the mounted path

Configuration
- FASTMCP_INDEXER_PORT (default 8001)
- HOST_INDEX_PATH bind-mounts the target repo into /work (read-only)

Add to your agent as a separate MCP endpoint (SSE):
- URL: http://localhost:8001/sse

Example calls (semantics vary by client):
- qdrant-index with args {"subdir":"scripts","recreate":true}

### MCP client configuration examples

Windsurf/Cursor (stdio for search + SSE for indexer):

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
      },
      "disabled": false
    },
    "qdrant-indexer": {
      "type": "sse",
      "url": "http://localhost:8001/sse",
      "disabled": false
    }
  }
}
```

Augment (SSE for both servers – recommended):

```json
{
  "mcpServers": {
    "qdrant": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
```

Qodo (Remote MCPs use simplified format - add each tool individually):

**Note**: In Qodo, you must add each MCP tool separately through the UI, not as a single JSON config.

For each tool, use this format:

**Tool 1 - qdrant:**
```json
{
  "qdrant": { "url": "http://localhost:8000/sse" }
}
```

**Tool 2 - qdrant-indexer:**
```json
{
  "qdrant-indexer": { "url": "http://localhost:8001/sse" }
}
```

#### Important for IDE agents (Cursor/Windsurf/Augment)
- Do not send null values to MCP tools. Omit the field or pass an empty string "" instead.
- qdrant-index examples:
  - {"subdir":"","recreate":false,"collection":"my-collection","repo_name":"workspace"}
  - {"subdir":"scripts","recreate":true}
- For indexing the repo root with no params, use the zero-arg tool `qdrant_index_root` (new) or call `qdrant-index` with `subdir:""`.


##### Zero-config search tool (new)
- repo_search: run code search without filters or config.
  - Structured fields supported (parity with DSL): language, under, kind, symbol, ext, not_, case, path_regex, path_glob, not_glob
  - Response shaping: compact (bool) returns only path/start_line/end_line
  - Smart default: compact=true when query is an array with multiple queries (unless explicitly set)
  - If include_snippet is true, compact is forced off so snippet fields are returned

  - Glob fields accept a single string or an array; you can also pass a comma-separated string which will be split
  - Query parsing: accepts query or queries; JSON arrays, JSON-stringified arrays, comma-separated strings; also supports q/text aliases

  - Parity note: path_glob/not_glob list handling works in both modes — in-process and subprocess — with OR semantics for path_glob and reject-on-any for not_glob.
  - Examples:
    - {"query": "semantic chunking"}
    - {"query": ["function to split code", "overlapping chunks"], "limit": 15, "per_path": 3}
    - {"query": "watcher debounce", "language": "python", "under": "scripts/", "include_snippet": true, "context_lines": 2}
    - {"query": "parser", "ext": "ts", "path_regex": "/services/.+", "compact": true}
    - {"query": "adapter", "path_glob": ["**/src/**", "**/pkg/**"], "not_glob": "**/tests/**"}
  - Returns structured results: score, path, symbol, start_line, end_line, and optional snippet; or compact form.
- code_search: alias of repo_search (same args) for easier discovery in some clients.

- qdrant_status: return collection size and last index times (safe, read-only).
  - {"collection": "my-collection"}


Verification:
- You should see tools from both servers (e.g., `store`, `find`, `repo_search`, `code_search`, `context_search`, `qdrant_list`, `qdrant_index`, `qdrant_prune`, `qdrant_status`).
- Call `qdrant_list` to confirm Qdrant connectivity.
- Call `qdrant_index` with args like `{ "subdir": "scripts", "recreate": true }` to (re)index the mounted repo.
- Call `context_search` with `{ "include_memories": true }` to blend memory+code (requires enabling MEMORY_SSE_ENABLED on the indexer service).

- qdrant_list with no args
- qdrant_prune with no args

```

Notes:
- The indexer reads env from `.env` (QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL).
- Default chunking: ~120 lines with 20-line overlap.
- Skips typical build/venv directories.
- Populates `metadata.kind`, `metadata.symbol`, and `metadata.symbol_path` for Python/JS/TS/Go/Java/Rust/Terraform (best-effort), per chunk.
- Uses the same collection as the MCP server.

### Exclusions (.qdrantignore) and defaults

- The indexer now supports a `.qdrantignore` file at the repo root (similar to `.gitignore`). Use it to exclude directories/files from indexing.
- Sensible defaults are excluded automatically (overridable): `/models`, `/node_modules`, `/dist`, `/build`, `/.venv`, `/venv`, `/__pycache__`, `/.git`, and files matching `*.onnx`, `*.bin`, `*.safetensors`, `tokenizer.json`, `*.whl`, `*.tar.gz`.
- Override via env or flags:
  - Env: `QDRANT_DEFAULT_EXCLUDES=0` to disable defaults; `QDRANT_IGNORE_FILE=.myignore`; `QDRANT_EXCLUDES='tokenizer.json,*.onnx,/third_party'`
  - CLI examples:
    - `docker compose run --rm indexer --root /work --ignore-file .qdrantignore`
    - `docker compose run --rm indexer --root /work --no-default-excludes --exclude '/vendor' --exclude '*.bin'`

### Scaling and tuning (small → large codebases)

- Chunking and batching are tunable via env or flags:
  - `INDEX_CHUNK_LINES` (default 120), `INDEX_CHUNK_OVERLAP` (default 20)
  - `INDEX_BATCH_SIZE` (default 64)
  - `INDEX_PROGRESS_EVERY` (default 200 files; 0 disables)
### Prune stale points (optional)

If files were deleted or significantly changed outside the indexer, remove stale points safely:

```bash
make prune
```

- CLI equivalents: `--chunk-lines`, `--chunk-overlap`, `--batch-size`, `--progress-every`.
- Recommendations:
  - Small repos (<100 files): chunk 80–120, overlap 16–24, batch-size 32–64
  - Medium (100s–1k files): chunk 120–160, overlap ~20, batch-size 64–128
  - Large monorepos (1k+): start with defaults; consider `INDEX_PROGRESS_EVERY=200` for visibility and `INDEX_BATCH_SIZE=128` if RAM allows

### MCP search filtering (language, path, kind)

- The indexer creates payload indexes for efficient filtering.
- When querying (via MCP client or scripts), you can filter by:
  - `metadata.language` (e.g., python, typescript, javascript, go, rust)
  - `metadata.path_prefix` (e.g., `/work/src`)
  - `metadata.kind` (e.g., function, class, method)
- Example: in the provided reranker script you can do:

```bash
make rerank ARGS="--language python --under /work/scripts"
```

- Direct Qdrant filter example is shown below; most MCP clients allow passing tool args that map to server-side filters. If your client supports adding structured args to `qdrant-find`, prefer these filters to reduce noise.


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
- `metadata.file_hash` (keyword)
- `metadata.ingested_at` (keyword)
- Git history fields available in payload: `commit_id`, `author_name`, `authored_date`, `message`, `files`

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

- If the MCP servers can’t reach Qdrant, confirm both containers are up: `make ps`.
- If the SSE port collides, change `FASTMCP_PORT` in `.env` and the mapped port in `docker-compose.yml`.
- If you customize tool descriptions, restart: `make restart`.

