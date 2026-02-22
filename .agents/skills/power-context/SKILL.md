---
name: power-context
description: Use this skill for local code indexing or code search. Covers indexing, auto-reindex watch mode, and code retrieval commands for implementations, symbols, tests, and config files.
---

# Power Context

Use `power-context` for code indexing and retrieval only.
Call CLI commands directly; do not call MCP tools.

## Prerequisites

- Qdrant reachable at `QDRANT_URL` (default `http://localhost:6333`)
- Python environment with project dependencies installed
- Collection selected with `--collection` or `COLLECTION_NAME`

## Command Surface

Use only these commands for this skill:

**Core Search Commands:**
- `search` - Hybrid semantic/lexical search with neural reranking
- `search-tests` - Find test files related to a query
- `search-config` - Find configuration files
- `search-callers` - Find callers of a symbol/function
- `search-importers` - Find importers of a module
- `symbol-graph` - Navigate symbol relationships (callers, definition, importers)
- `pattern-search` - Structural code pattern matching

**Index Lifecycle Commands:**
- `index` - Index a directory into Qdrant
- `watch` - Auto-reindex on file changes (daemon)
- `prune` - Remove stale/deleted files from index
- `status` - Collection status and health
- `list-collections` - List all Qdrant collections

## Operating Workflow

1. Ensure target collection exists or initialize via `index`.
2. For active projects, run `watch` to auto-reindex file changes.
3. Use `search*`/`symbol-graph`/`pattern-search` for retrieval.
4. Use `status` or `list-collections` for collection diagnostics.
5. Use `prune` to remove stale points after large deletions/moves.

All command results are JSON on stdout. Treat stderr as logs.

## Examples



### Search
```bash
# Basic search
power-context search "authentication middleware"

# Search with filters
power-context search "auth middleware" --language python --under src/

# Find test files
power-context search-tests "UserService" --language python

# Find config files
power-context search-config "database url"

# Find symbol callers
power-context search-callers "processPayment"

# Find module importers
power-context search-importers "qdrant_client"

# Navigate symbol graph
power-context symbol-graph ASTAnalyzer --query-type definition

# Structural pattern search
power-context pattern-search "if err != nil { return err }" --language go
```

### Indexing
```bash
# Index current directory
power-context index . --collection codebase

# Index with recreation (drops existing data)
power-context index . --collection codebase --recreate

# Start watch daemon for auto-reindexing
power-context watch . --collection codebase
```
### Index Maintenance
```bash
# Remove stale entries
power-context prune . --collection codebase

# Check collection status
power-context status --collection codebase

# List all collections
power-context list-collections
```

## Environment Variables

`QDRANT_URL`, `COLLECTION_NAME`, `DEFAULT_COLLECTION`, `QDRANT_TIMEOUT`, `EMBEDDING_MODEL`
