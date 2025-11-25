This file is intended for AI agents (Claude, etc.) using the Context‑Engine Qdrant‑Indexer and Memory MCP tools. It encodes project‑specific best practices; adapt it per‑repo.


Agentic AI Project Rules: When to Use MCP Qdrant-Indexer vs Grep

  Core Decision Rules (for AI agents)

  - Use MCP Qdrant-Indexer when:
    - You are exploring or don't know exact strings/symbols.
    - You need semantic or cross-file understanding (relationships, patterns, architecture).
    - You want ranked results with surrounding context, not just line hits.

  - Use grep when:
    - You know the exact string/function/variable or error message.
    - You need fast literal search or are extremely token/latency constrained.

  Quick Heuristics:

  - If you know the exact string → start with grep, then switch to MCP for broader context.
  - If the question is conceptual/architectural → start with MCP.
  - If you need rich context/snippets around matches → MCP.
  - If you just need to confirm existence/location → grep.

  Grep Anti-Patterns:

  # DON'T - Wasteful when semantic search needed
  grep -r "auth" .                    # → Use MCP: "authentication mechanisms"
  grep -r "cache" .                   # → Use MCP: "caching strategies"
  grep -r "error" .                   # → Use MCP: "error handling patterns"
  grep -r "database" .                # → Use MCP: "database operations"

  # DO - Efficient for exact matches
  grep -rn "UserAlreadyExists" .      # Specific error class
  grep -rn "def authenticate_user" .  # Exact function name
  grep -rn "REDIS_HOST" .            # Exact environment variable

  MCP Tool Patterns:

  # DO - Use concept/keyword-style queries (short natural-language fragments)
  "input validation mechanisms"
  "database connection handling"
  "performance bottlenecks in request path"
  "places where user sessions are managed"
  "logging and error reporting patterns"

  MCP Qdrant-Indexer Specific Knobs

  Essential Parameters:

  - limit: Control result count (3-8 for efficiency)
  - per_path: Limit results per file (1-2 prevents redundancy)
  - compact=true: Reduces token usage by 60-80%
  - include_snippet=false: Headers only when speed matters
  - collection: Target specific codebases for precision

  Performance Optimization:

  - Start with limit=3, compact=true for discovery
  - Increase to limit=5, include_snippet=true for details
  - Use language and under filters to narrow scope
  - Set rerank_enabled=false for faster but less accurate results

  When to Use Advanced Features:

  - rerank_enabled=true: For complex queries needing best relevance
  - context_lines=5+: When you need implementation details
  - multiple collections: Cross-repo architectural analysis
  - symbol filtering: When looking for specific function/class types

  Anti-Patterns to Avoid:

  - Don't use limit=20 with include_snippet=true (token waste)
  - Don't search without collection specification (noise)
  - Don't ignore per_path limits (duplicate results from same file)
  - Don't use context lines for pure discovery (unnecessary tokens)

  Tool Roles Cheat Sheet:

  - repo_search / code_search:
    - Use for: finding relevant files/spans and inspecting raw code.
    - Think: "where is X implemented?", "show me usages of Y".
  - context_search:
    - Use for: combining code hits with memory/docs when both matter.
    - Good for: "give me related code plus any notes/docs I wrote".
  - context_answer:
    - Use for: natural-language explanations grounded in code, with citations.
    - Good for: "how do uploads get triggered when files change?", "where is the watcher wired into the indexer?".

  Query Phrasing Tips for context_answer:

  - Prefer behavior/architecture questions:
    - "How do uploads get triggered when files change?"
    - "Where is the VS Code file watcher that triggers indexing uploads?"
  - If you care about a specific file, mention it explicitly:
    - "What does ingest_code.py do?", "Explain ensureIndexedWatcher in extension.js".
  - Mentioning a specific filename can bias retrieval to that file; for cross-file wiring
    questions, prefer behavior-describing queries without filenames.
  - For very cross-file questions, you can:
    - First use repo_search to discover key files,
    - Then call context_answer with a behavior-focused question that doesn't over-specify filenames.

  Remember: the MCP tools themselves expose detailed descriptions and parameter docs.
  Use those for exact knobs; this guide is about choosing the right tool and shaping good queries.

  MCP Tool Families (for AI agents)

  - Indexer / Qdrant tools:
    - qdrant_index_root, qdrant_index, qdrant_prune
    - qdrant_list, qdrant_status
    - workspace_info, list_workspaces, collection_map
    - set_session_defaults
  - Search / QA tools:
    - repo_search, code_search, context_search, context_answer
    - search_tests_for, search_config_for, search_callers_for, search_importers_for
    - change_history_for_path, expand_query
  - Memory tools:
    - memory.set_session_defaults, memory.store, memory.find

  Additional behavioral tips:

  - Call set_session_defaults (indexer and memory) early in a session so subsequent
    calls inherit the right collection without repeating it in every request.
  - Use context_search with include_memories and per_source_limits when you want
    blended code + memory results instead of calling repo_search and memory.find
    separately.
  - Treat expand_query and the expand flag on context_answer as expensive options:
    only use them after a normal search/answer attempt failed to find good context.