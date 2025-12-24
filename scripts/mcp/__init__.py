"""
MCP (Model Context Protocol) indexer server package.

This package contains extracted modules from mcp_indexer_server.py:
- utils: Type coercion, JSON parsing, tokenization, env helpers
- toon: TOON output format support
- workspace: Workspace state and collection resolution
- admin_tools: Qdrant admin operations (index, list, status, prune)
- code_signals: Code intent detection
- context_answer: LLM-assisted Q&A with retrieval
- context_search: Blended code + memory search
- query_expand: LLM-assisted query expansion

Usage:
    from scripts.mcp import utils, toon, workspace
    from scripts.mcp.utils import _coerce_bool, _env_overrides
    from scripts.mcp.workspace import _default_collection
    from scripts.mcp.context_search import _context_search_impl
    from scripts.mcp.query_expand import _expand_query_impl
"""
from scripts.mcp import utils
from scripts.mcp import toon
from scripts.mcp import workspace
from scripts.mcp import admin_tools
from scripts.mcp import code_signals
from scripts.mcp import context_answer
from scripts.mcp import context_search
from scripts.mcp import query_expand
from scripts.mcp import search
from scripts.mcp import info_request
from scripts.mcp import memory
from scripts.mcp import search_specialized
from scripts.mcp import search_history

__all__ = [
    "utils",
    "toon",
    "workspace",
    "admin_tools",
    "code_signals",
    "context_answer",
    "context_search",
    "query_expand",
    "search",
    "info_request",
    "memory",
    "search_specialized",
    "search_history",
]

