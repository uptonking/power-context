# MCP API Reference

This document provides comprehensive API documentation for all MCP (Model Context Protocol) tools exposed by Context Engine's dual-server architecture.

**Documentation:** [README](../README.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Overview](#overview)
- [Memory Server API](#memory-server-api) - `store()`, `find()`
- [Indexer Server API](#indexer-server-api) - `repo_search()`, `context_search()`, `context_answer()`, etc.
- [Response Schemas](#response-schemas)
- [Error Handling](#error-handling)

---

## Overview

Context Engine exposes two MCP servers:

1. **Memory Server**: Knowledge base storage and retrieval (`port 8000` SSE, `port 8002` HTTP)
2. **Indexer Server**: Code search, indexing, and management (`port 8001` SSE, `port 8003` HTTP)

Both servers support SSE and HTTP RMCP transports simultaneously.

## Memory Server API

### store()

Store information with rich metadata for later retrieval and search.

**Parameters:**
- `information` (str, required): Clear natural language description of the content to store
- `metadata` (dict, optional): Structured metadata with the following schema:
  - `kind` (str, optional): Category type - one of:
    - `"snippet"`: Code snippet or pattern
    - `"explanation"`: Technical explanation
    - `"pattern"`: Design pattern or approach
    - `"example"`: Usage example
    - `"reference"`: Reference information
  - `language` (str, optional): Programming language (e.g., "python", "javascript", "go")
  - `path` (str, optional): File path context for code-related entries
  - `tags` (list[str], optional): Searchable tags for categorization
  - `priority` (int, optional): Importance ranking (1-10, higher = more important)
  - `topic` (str, optional): High-level topic classification
  - `code` (str, optional): Actual code content (for snippet kind)
  - `author` (str, optional): Author or source attribution
  - `created_at` (str, optional): ISO timestamp (auto-generated if omitted)

**Returns:**
```json
{
  "ok": true,
  "id": "uuid-string",
  "message": "Successfully stored information"
}
```

**Example:**
```json
{
  "information": "Efficient Python pattern for processing large files using generators to minimize memory usage",
  "metadata": {
    "kind": "pattern",
    "language": "python",
    "path": "utils/file_processor.py",
    "tags": ["python", "generators", "memory-efficient", "performance"],
    "priority": 8,
    "topic": "performance optimization",
    "code": "def process_large_file(file_path):\n    with open(file_path) as f:\n        for line in f:\n            yield process_line(line)"
  }
}
```

### find()

Search stored memories using hybrid retrieval (semantic + lexical search).

**Parameters:**
- `query` (str, required): Search query or question
- `kind` (str, optional): Filter by entry kind (snippet, explanation, pattern, etc.)
- `language` (str, optional): Filter by programming language
- `topic` (str, optional): Filter by topic
- `tags` (str or list[str], optional): Filter by tags (comma-separated string or list)
- `limit` (int, default 10): Maximum number of results to return
- `priority_min` (int, optional): Minimum priority threshold (1-10)

**Returns:**
```json
{
  "ok": true,
  "results": [
    {
      "id": "uuid-string",
      "information": "Full stored information text",
      "metadata": {
        "kind": "pattern",
        "language": "python",
        "path": "utils/file_processor.py",
        "tags": ["python", "generators"],
        "priority": 8,
        "topic": "performance",
        "created_at": "2024-01-15T10:30:00Z"
      },
      "score": 0.89,
      "highlights": ["<<efficient>> Python pattern", "<<memory usage>>"]
    }
  ],
  "total": 15,
  "query": "python file processing generators"
}
```

**Example:**
```json
{
  "query": "database connection pooling patterns",
  "language": "python",
  "kind": "pattern",
  "limit": 5
}
```

## Indexer Server API

### repo_search()

Perform hybrid code search combining dense semantic, lexical BM25, and optional neural reranking.

**Core Parameters:**
- `query` (str or list[str], required): Search query or list of queries for query fusion
- `limit` (int, default 10): Maximum total results to return
- `per_path` (int, default 2): Maximum results per file path

**Content Filters:**
- `language` (str, optional): Filter by programming language
- `path_glob` (str or list[str], optional): Glob patterns for path filtering
- `under` (str, optional): Limit search to specific directory path
- `not_glob` (str or list[str], optional): Exclude paths matching these patterns

**Code Structure Filters:**
- `symbol` (str, optional): Search for specific function, class, or variable names
- `kind` (str, optional): Filter by code construct type:
  - `"function"`: Function definitions
  - `"class"`: Class definitions
  - `"variable"`: Variable assignments
  - `"import"`: Import statements
  - `"comment"`: Comments and docstrings

**Search Options:**
- `include_snippet` (bool, default true): Include code snippet in results
- `context_lines` (int, default 3): Number of context lines around snippet
- `highlight_snippet` (bool, default true): Highlight matching tokens in snippet

**Reranking Options:**
- `rerank_enabled` (bool, optional): Override default reranker setting
- `rerank_top_n` (int, default 50): Number of candidates to consider for reranking
- `rerank_return_m` (int, default 12): Number of results to return after reranking

**Response Format:**
```json
{
  "ok": true,
  "results": [
    {
      "score": 0.89,
      "path": "src/search/hybrid_search.py",
      "symbol": "hybrid_search",
      "start_line": 45,
      "end_line": 67,
      "snippet": "def hybrid_search(query, limit=10):\n    # ReFRAG-inspired implementation\n    results = []\n    return results",
      "highlights": ["<<ReFRAG-inspired>> implementation"],
      "components": {
        "dense_score": 0.85,
        "lexical_score": 0.42,
        "reranker_score": 0.91,
        "final_score": 0.89
      },
      "metadata": {
        "language": "python",
        "kind": "function",
        "complexity": "medium",
        "tokens": 156
      }
    }
  ],
  "total": 15,
  "used_rerank": true,
  "search_time_ms": 127,
  "query": "asyncio subprocess management python"
}
```

**Examples:**

**Basic Search:**
```json
{
  "query": "asyncio subprocess management",
  "limit": 10,
  "language": "python"
}
```

**Advanced Search with Multiple Filters:**
```json
{
  "query": ["database connection", "sqlalchemy pool"],
  "language": "python",
  "path_glob": "**/db/**/*.py",
  "not_glob": ["**/test_*.py", "**/migrations/**"],
  "kind": "function",
  "limit": 20,
  "per_path": 3,
  "rerank_enabled": true
}
```

**Symbol Search:**
```json
{
  "query": "hybrid_search",
  "symbol": "hybrid_search",
  "language": "python",
  "include_snippet": true
}
```

### context_search()

Blend code search results with memory entries for comprehensive context.

**Parameters:**
All `repo_search` parameters plus:
- `include_memories` (bool, default true): Whether to include memory results
- `memory_weight` (float, default 1.0): Weight for memory results vs code results
- `per_source_limits` (dict, optional): Limits per source type:
  ```json
  {
    "code": 8,
    "memory": 4
  }
  ```

**Returns:**
```json
{
  "ok": true,
  "results": [
    {
      "source": "code",
      "score": 0.89,
      "path": "src/db/connection.py",
      "symbol": "create_pool",
      "snippet": "def create_pool(database_url):\n    return create_engine(database_url, pool_size=10)"
    },
    {
      "source": "memory",
      "score": 0.85,
      "id": "uuid-string",
      "information": "Database connection pooling best practices for high-concurrency applications",
      "metadata": {
        "kind": "pattern",
        "language": "python",
        "priority": 9
      }
    }
  ],
  "total": 12,
  "sources": ["code", "memory"],
  "query": "database connection pooling"
}
```

### context_answer()

Generate natural language answers using retrieval-augmented generation with local LLM.

**Core Parameters:**
- `query` (str or list[str], required): Question or query to answer
- `budget_tokens` (int, optional): Token budget for context assembly (default from config)
- `include_snippet` (bool, default true): Include code snippets in context

**Retrieval Parameters:**
All `repo_search` parameters supported for context retrieval.

**LLM Parameters:**
- `max_tokens` (int, optional): Maximum tokens in generated answer
- `temperature` (float, default 0.3): Sampling temperature (lower = more deterministic)
- `mode` (str, default "stitch"): Context assembly mode ("stitch" or "pack")
- `expand` (bool, default false): Enable query expansion

**Response Format:**
```json
{
  "ok": true,
  "answer": "Context Engine uses ReFRAG-inspired micro-chunking with 16-token windows and 8-token stride to achieve precise code retrieval. The span budgeting system ensures efficient token usage while maintaining context relevance.",
  "citations": [
    {
      "path": "scripts/hybrid_search.py",
      "start_line": 156,
      "end_line": 162,
      "snippet": "# ReFRAG micro-chunking\nWINDOW_SIZE = 16\nSTRIDE = 8",
      "relevance": 0.92
    },
    {
      "path": "scripts/utils.py",
      "start_line": 89,
      "end_line": 95,
      "snippet": "def micro_chunk(text, window_size=16, stride=8):",
      "relevance": 0.87
    }
  ],
  "query": ["How does Context Engine implement micro-chunking?"],
  "used_context_tokens": 1247,
  "generation_time_ms": 2340,
  "decoder_used": "llamacpp"
}
```

**Example:**
```json
{
  "query": "What is the best way to handle database connections in Python web applications?",
  "budget_tokens": 2000,
  "language": "python",
  "expand": true,
  "temperature": 0.2
}
```

### qdrant_index()

Index or reindex code from the mounted workspace.

**Parameters:**
- `subdir` (str, optional): Subdirectory to index (default: entire workspace)
- `recreate` (bool, default false): Drop and recreate collection before indexing
- `collection` (str, optional): Override default collection name

**Returns:**
```json
{
  "ok": true,
  "operation": "index",
  "subdir": "",
  "collection": "my-workspace",
  "recreate": false,
  "stats": {
    "files_processed": 1250,
    "chunks_created": 8432,
    "vectors_generated": 8432,
    "processing_time_seconds": 127,
    "errors": 0
  },
  "message": "Indexing completed successfully"
}
```

### qdrant_prune()

Remove stale points from the collection (files that no longer exist).

**Parameters:** None (operates on current workspace)

**Returns:**
```json
{
  "ok": true,
  "operation": "prune",
  "points_removed": 47,
  "points_before": 15234,
  "points_after": 15187,
  "processing_time_ms": 892,
  "message": "Pruning completed successfully"
}
```

### qdrant_status()

Get comprehensive status information about the collection and indexing state.

**Parameters:**
- `collection` (str, optional): Override default collection name
- `max_points` (int, default 5000): Maximum points to scan for timestamp analysis
- `batch` (int, default 1000): Batch size for scanning

**Returns:**
```json
{
  "ok": true,
  "collection": "my-workspace",
  "exists": true,
  "count": 15234,
  "scanned_points": 5000,
  "last_ingested_at": {
    "unix": 1705123456,
    "iso": "2024-01-13T15:30:56Z"
  },
  "last_modified_at": {
    "unix": 1705124123,
    "iso": "2024-01-13T15:35:23Z"
  },
  "vectors_config": {
    "fast-bge-base-en-v1.5": 384,
    "lex": 4096
  },
  "storage_size_mb": 245.7,
  "status": "healthy"
}
```

### qdrant_list()

List all available Qdrant collections.

**Parameters:** None

**Returns:**
```json
{
  "ok": true,
  "collections": [
    {
      "name": "my-workspace",
      "vectors_count": 15234,
      "segments_count": 12,
      "points_count": 15234,
      "indexed_vectors_count": 15234,
      "status": "green",
      "optimizer_status": "ok"
    }
  ]
}
```

### workspace_info()

Read workspace state and default collection information.

**Parameters:**
- `workspace_path` (str, optional): Override workspace path (default: current workspace)

**Returns:**
```json
{
  "ok": true,
  "workspace_path": "/work",
  "default_collection": "context-engine-workspace",
  "source": "state_file",
  "state": {
    "workspace_id": "workspace-uuid",
    "created_at": "2024-01-10T09:15:00Z",
    "last_indexed": "2024-01-13T15:30:56Z",
    "files_count": 1250,
    "total_size_bytes": 52428800
  }
}
```

### list_workspaces()

Scan for all workspaces with .codebase/state.json files.

**Parameters:**
- `search_root` (str, optional): Root directory to scan (default: parent of workspace)

**Returns:**
```json
{
  "ok": true,
  "workspaces": [
    {
      "workspace_path": "/work",
      "collection_name": "context-engine-workspace",
      "last_updated": "2024-01-13T15:30:56Z",
      "indexing_state": "completed"
    },
    {
      "workspace_path": "/work/project-b",
      "collection_name": "project-b-workspace",
      "last_updated": "2024-01-12T11:20:30Z",
      "indexing_state": "in_progress"
    }
  ]
}
```

### memory_store()

Store memory entry (alias for Memory Server's `store()` tool).

**Parameters:** Same as Memory Server `store()` method

**Returns:** Same as Memory Server `store()` method

### expand_query()

Generate alternative query variations using local LLM (requires decoder enabled).

**Parameters:**
- `query` (str or list[str], required): Original query or queries to expand
- `max_new` (int, default 2): Maximum number of alternative queries to generate

**Returns:**
```json
{
  "ok": true,
  "original_query": "python asyncio subprocess",
  "alternates": [
    "python asynchronous process management",
    "asyncio subprocess handling in python"
  ],
  "total_queries": 3,
  "decoder_used": "llamacpp"
}
```

### code_search()

Exact alias of `repo_search()` for discoverability. Same parameters and return shape.

### qdrant_index_root()

Index the entire workspace root (`/work`).

**Parameters:**
- `recreate` (bool, default false): Drop and recreate collection before indexing
- `collection` (str, optional): Target collection name

**Returns:** Subprocess result with indexing status.

### search_tests_for()

Find test files related to a query. Presets common test file globs.

**Parameters:**
- `query` (str or list[str], required): Search query
- `limit` (int, optional): Max results
- `include_snippet` (bool, optional): Include code snippets
- `language` (str, optional): Filter by language

**Returns:** Same shape as `repo_search()`.

### search_config_for()

Find configuration files related to a query. Presets config file globs (yaml/json/toml/etc).

**Parameters:** Same as `search_tests_for()`.

**Returns:** Same shape as `repo_search()`.

### search_callers_for()

Heuristic search for callers/usages of a symbol.

**Parameters:**
- `query` (str, required): Symbol name to find callers for
- `limit` (int, optional): Max results
- `language` (str, optional): Filter by language

**Returns:** Same shape as `repo_search()`.

### search_importers_for()

Find files likely importing or referencing a module/symbol.

**Parameters:** Same as `search_callers_for()`.

**Returns:** Same shape as `repo_search()`.

### change_history_for_path()

Summarize recent change metadata for a file path from the index.

**Parameters:**
- `path` (str, required): Relative path under /work
- `collection` (str, optional): Target collection
- `max_points` (int, optional): Cap on scanned points

**Returns:**
```json
{
  "ok": true,
  "summary": {
    "path": "scripts/ctx.py",
    "last_modified": "2025-01-15T14:22:00"
  }
}
```

### collection_map()

Return collection↔repo mappings with optional Qdrant payload samples.

**Parameters:**
- `search_root` (str, optional): Directory to scan
- `collection` (str, optional): Filter by collection
- `repo_name` (str, optional): Filter by repo
- `include_samples` (bool, optional): Include payload samples
- `limit` (int, optional): Max entries

**Returns:** Mapping of collections to repositories.

### set_session_defaults() (Indexer)

Set default collection for subsequent calls on the same session.

**Parameters:**
- `collection` (str, optional): Default collection name
- `session` (str, optional): Session token for cross-connection reuse

**Returns:**
```json
{
  "ok": true,
  "session": "abc123",
  "defaults": {"collection": "codebase"},
  "applied": "connection"
}
```

## Error Handling

All API methods follow consistent error handling patterns:

### Standard Error Response
```json
{
  "ok": false,
  "error": "Error type and description",
  "error_code": "VALIDATION_ERROR",
  "details": {
    "field": "query",
    "message": "Query cannot be empty"
  }
}
```

### Common Error Codes
- `VALIDATION_ERROR`: Invalid parameter values
- `COLLECTION_NOT_FOUND`: Specified collection doesn't exist
- `INDEXING_ERROR`: Failed during indexing operation
- `SEARCH_ERROR`: Search operation failed
- `DECODER_ERROR`: LLM decoder operation failed
- `TIMEOUT_ERROR`: Operation timed out
- `RATE_LIMIT_ERROR`: Too many requests

## Rate Limits and Quotas

- **Default timeout**: 30 seconds per operation
- **Maximum query length**: 1000 characters
- **Maximum result limit**: 100 results per search
- **Memory storage**: Configurable per deployment
- **Batch indexing limits**: Configurable via environment variables

## Transport-Specific Behavior

### SSE (Server-Sent Events)
- Real-time bidirectional communication
- Automatic reconnection on disconnect
- Streaming responses for long operations

### HTTP RMCP
- JSON-RPC over HTTP
- Request/response pattern
- Better for batch operations and integrations

Both transports provide identical API semantics and response formats.

This API reference should enable developers to effectively integrate Context Engine's MCP tools into their applications and workflows.