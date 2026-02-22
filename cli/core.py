"""Shared helpers for CLI commands."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path (fallback for development mode)
try:
    import scripts
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
DEFAULT_COLLECTION = (
    os.environ.get("DEFAULT_COLLECTION")
    or os.environ.get("COLLECTION_NAME")
    or "codebase"
)

# Lazy singletons
_client = None
_model = None
_model_dim = None


def get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(
            url=QDRANT_URL,
            timeout=int(os.environ.get("QDRANT_TIMEOUT", "20")),
        )
    return _client


def get_model():
    global _model
    if _model is None:
        from scripts.embedder import get_embedding_model
        _model = get_embedding_model(MODEL_NAME)
    return _model


def get_model_dim() -> int:
    global _model_dim
    if _model_dim is None:
        from scripts.embedder import get_model_dimension
        _model_dim = get_model_dimension(MODEL_NAME)
    return _model_dim


def resolve_collection(
    override: str | None = None,
    *,
    workspace_path: str | Path | None = None,
) -> str:
    """Resolve collection: CLI arg > env > workspace state > default.

    When workspace_path is provided, prefer workspace_state resolution for that path.
    """
    if override:
        return override
    if DEFAULT_COLLECTION != "codebase":
        return DEFAULT_COLLECTION
    try:
        from scripts.workspace_state import (
            get_collection_name_with_staging as _get_collection_name_with_staging,
            is_multi_repo_mode as _is_multi_repo_mode,
        )

        if workspace_path is not None and _is_multi_repo_mode():
            ws_path: str | None = str(Path(workspace_path).resolve())
        else:
            ws_path = None
        coll = _get_collection_name_with_staging(ws_path)
        if coll:
            return str(coll)
    except Exception:
        pass
    return DEFAULT_COLLECTION


def output_json(data: Any) -> None:
    """Write JSON to stdout — single place for all commands."""
    json.dump(data, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def run_async(coro) -> Any:
    """Run an async coroutine from sync CLI context."""
    return asyncio.run(coro)


async def repo_search_async(**kwargs) -> Dict[str, Any]:
    """Wired repo_search that specialized _impl functions can use as repo_search_fn.

    This mirrors what mcp_indexer_server.py passes as repo_search_fn — a fully
    wired async function that the specialized search impls delegate to.
    """
    from scripts.mcp_impl.search import _repo_search_impl
    from scripts.mcp_impl.admin_tools import _get_embedding_model, _run_async

    return await _repo_search_impl(
        **kwargs,
        get_embedding_model_fn=_get_embedding_model,
        require_auth_session_fn=lambda s: None,  # no auth in CLI
        do_highlight_snippet_fn=None,
        run_async_fn=_run_async,
    )
