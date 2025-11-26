# Troubleshooting Guide

Common issues and solutions for Context Engine.

**Documentation:** [README](../README.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Collection Health & Cache Sync](#collection-health--cache-sync)
- [Common Issues](#common-issues)
- [Connectivity Issues](#connectivity-issues)
- [Endpoint Verification](#endpoint-verification)
- [Debug Logging](#debug-logging)

---

## Collection Health & Cache Sync

The stack includes automatic health checks that detect and fix cache/collection sync issues.

### Check collection health
```bash
python scripts/collection_health.py --workspace . --collection codebase
```

### Auto-heal cache issues
```bash
python scripts/collection_health.py --workspace . --collection codebase --auto-heal
```

### What it detects
- Empty collection with cached files (cache thinks files are indexed but they're not)
- Significant mismatch between cached files and actual collection contents
- Missing metadata in collection points

### When to use
- After manually deleting collections
- If searches return no results despite indexing
- After Qdrant crashes or data loss
- When switching between collection names

### Automatic healing
- Health checks run automatically on watcher and indexer startup
- Cache is cleared when sync issues are detected
- Files are reindexed on next run

---

## Common Issues

### Tree-sitter not found or parser errors
Feature is optional. If you set `USE_TREE_SITTER=1` and see errors, unset it or install tree-sitter deps, then reindex.

### Tokenizer missing for micro-chunks
Run `make tokenizer` or set `TOKENIZER_JSON` to a valid tokenizer.json. Otherwise, falls back to line-based chunking.

### SSE "Invalid session ID" when POSTing /messages directly
Expected if you didn't initiate an SSE session first. Use an MCP client (e.g., mcp-remote) to handle the handshake.

### llama.cpp platform warning on Apple Silicon
Prefer the native path (`scripts/gpu_toggle.sh gpu`). If you stick with Docker, add `platform: linux/amd64` to the service or ignore the warning during local dev.

### Indexing feels stuck on very large files
Use `MAX_MICRO_CHUNKS_PER_FILE=200` during dev runs.

### Watcher timeouts (-9) or Qdrant "ResponseHandlingException: timed out"
Set watcher-safe defaults to reduce payload size and add headroom during upserts:

```ini
QDRANT_TIMEOUT=60
MAX_MICRO_CHUNKS_PER_FILE=200
INDEX_UPSERT_BATCH=128
INDEX_UPSERT_RETRIES=5
INDEX_UPSERT_BACKOFF=0.5
WATCH_DEBOUNCE_SECS=1.5
```

If issues persist, try lowering `INDEX_UPSERT_BATCH` to 96 or raising `QDRANT_TIMEOUT` to 90.

---

## Connectivity Issues

### MCP servers can't reach Qdrant
Confirm both containers are up: `make ps`.

### SSE port collides
Change `FASTMCP_PORT` in `.env` and the mapped port in `docker-compose.yml`.

### Searches return no results
Check collection health (see above).

### Tool descriptions out of date
Restart: `make restart`.

---

## Verify Endpoints

```bash
# Qdrant DB
curl -sSf http://localhost:6333/readyz >/dev/null && echo "Qdrant OK"

# Decoder (llama.cpp sidecar)
curl -s http://localhost:8080/health

# SSE endpoints (Memory, Indexer)
curl -sI http://localhost:8000/sse | head -n1
curl -sI http://localhost:8001/sse | head -n1

# RMCP endpoints (HTTP JSON-RPC)
curl -sI http://localhost:8002/mcp | head -n1
curl -sI http://localhost:8003/mcp | head -n1
```

---

## Expected HTTP Behaviors

- **GET /mcp returns 400**: Normal - the RMCP endpoint is POST-only for JSON-RPC
- **SSE requires session handshake**: Raw POST /messages without it will error (expected)

---

## Operational Safeguards

| Setting | Purpose | Default |
|---------|---------|---------|
| TOKENIZER_JSON | Tokenizer for micro-chunking | models/tokenizer.json |
| MAX_MICRO_CHUNKS_PER_FILE | Prevent runaway chunk counts | 2000 |
| QDRANT_TIMEOUT | HTTP timeout for MCP Qdrant calls | 20s |
| MEMORY_AUTODETECT | Auto-detect memory collection | 1 |
| MEMORY_COLLECTION_TTL_SECS | Cache TTL for collection detection | 300s |

**Schema repair:** `ensure_collection` now repairs missing named vectors (lex, mini when REFRAG_MODE=1) on existing collections.

---

## Debug Logging

Enable debug environment variables for detailed logging:

```bash
export DEBUG_CONTEXT_ANSWER=1
export HYBRID_DEBUG=1
export CACHE_DEBUG=1

# Restart services
docker-compose restart
```

---

## Getting Help

1. Check this troubleshooting guide
2. Review logs: `docker compose logs mcp_indexer`
3. Verify health: `make health`
4. Check Qdrant status: `make qdrant-status`

