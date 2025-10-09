#!/usr/bin/env python3
"""
Minimal MCP (SSE) companion server exposing:
- qdrant-list: list collections
- qdrant-index: index the currently mounted path (/work or /work/<subdir>)
- qdrant-prune: prune stale points for the mounted path

This server is designed to run in a Docker container with the repository
bind-mounted at /work (read-only is fine). It reuses the same Python deps as the
indexer image and shells out to our existing scripts to keep behavior consistent.

Environment:
- FASTMCP_HOST (default: 0.0.0.0)
- FASTMCP_INDEXER_PORT (default: 8001)
- QDRANT_URL (e.g., http://qdrant:6333)
- COLLECTION_NAME (default: my-collection)
- REPO_NAME (default: workspace)

Note: We use the fastmcp library for quick SSE hosting. If you change to another
MCP server framework, keep the tool names and args stable.
"""
from __future__ import annotations
import json
import os
import subprocess
from typing import Any, Dict, Optional

try:
    # Official MCP Python SDK (FastMCP convenience server)
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    raise SystemExit("mcp package is required inside the container: pip install mcp")

APP_NAME = os.environ.get("FASTMCP_SERVER_NAME", "qdrant-indexer-mcp")
HOST = os.environ.get("FASTMCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("FASTMCP_INDEXER_PORT", "8001"))

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
DEFAULT_REPO = os.environ.get("REPO_NAME", "workspace")

mcp = FastMCP(APP_NAME)


def _run(cmd: list[str], env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return {
        "ok": proc.returncode == 0,
        "code": proc.returncode,
        "stdout": proc.stdout[-4000:],  # trim to avoid giant payloads
        "stderr": proc.stderr[-4000:],
    }


@mcp.tool()
async def qdrant_index_root(recreate: bool = False,
                            collection: str | None = None,
                            repo_name: str | None = None) -> Dict[str, Any]:
    """Index the mounted root path (/work) with zero-arg safe defaults.
    Notes for IDE agents (Cursor/Windsurf/Augment):
    - Prefer this tool when you want to index the repo root without specifying params.
    - Do NOT send null values to tools; either omit a field or pass an empty string "".
    - Args:
      - recreate (bool, default false): drop and recreate collection schema if needed
      - collection (string, optional): defaults to env COLLECTION_NAME
      - repo_name (string, optional): defaults to env REPO_NAME
    """
    coll = collection or DEFAULT_COLLECTION
    repo = repo_name or DEFAULT_REPO

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll
    env["REPO_NAME"] = repo

    cmd = ["python", "/work/scripts/ingest_code.py", "--root", "/work"]
    if recreate:
        cmd.append("--recreate")
    res = _run(cmd, env=env)
    return {"args": {"root": "/work", "collection": coll, "repo_name": repo, "recreate": recreate}, **res}

@mcp.tool()

async def qdrant_list() -> Dict[str, Any]:
    """List Qdrant collections"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
        cols = client.get_collections().collections
        return {"collections": [c.name for c in cols]}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def qdrant_index(subdir: Optional[str] = None, recreate: bool = False,
                 collection: Optional[str] = None, repo_name: Optional[str] = None) -> Dict[str, Any]:
    """Index the mounted path (/work) or a subdirectory.
    Important for IDE agents (Cursor/Windsurf/Augment):
    - Do NOT pass null values; omit a field or pass empty string "".
    - subdir: "" or omit to index repo root; or a relative path like "scripts"
    - recreate: bool (default false)
    - collection: string (optional; defaults to env COLLECTION_NAME)
    - repo_name: string (optional; defaults to env REPO_NAME)
    """
    root = "/work"
    if subdir:
        subdir = subdir.lstrip("/")
        root = os.path.join(root, subdir)
    coll = collection or DEFAULT_COLLECTION
    repo = repo_name or DEFAULT_REPO

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll
    env["REPO_NAME"] = repo

    cmd = [
        "python", "/work/scripts/ingest_code.py", "--root", root,
    ]
    if recreate:
        cmd.append("--recreate")
    res = _run(cmd, env=env)
    return {"args": {"root": root, "collection": coll, "repo_name": repo, "recreate": recreate}, **res}


@mcp.tool()
async def qdrant_prune() -> Dict[str, Any]:
    """Prune stale points for the mounted path (/work)"""
    env = os.environ.copy()
    env["PRUNE_ROOT"] = "/work"
    cmd = ["python", "/work/scripts/prune.py"]
    res = _run(cmd, env=env)
    return res

@mcp.tool()
async def repo_search(query, limit: int = 10, per_path: int = 2) -> Dict[str, Any]:
    """Zero-config code search over the mounted repo via Qdrant using hybrid_search defaults.
    Args:
      - query: string or list of strings
      - limit: total number of results to return (default 10)
      - per_path: cap of results per file path to diversify output (default 2)
    Notes:
      - No filters required; uses existing environment defaults (COLLECTION_NAME, QDRANT_URL).
      - Returns structured results parsed from hybrid_search output.
    """
    # Normalize queries to a list[str]
    queries: list[str] = []
    if isinstance(query, str):
        if query.strip():
            queries = [query]
    elif isinstance(query, (list, tuple)):
        for q in query:
            qs = str(q).strip()
            if qs:
                queries.append(qs)
    else:
        qs = str(query).strip()
        if qs:
            queries = [qs]

    if not queries:
        return {"error": "query required"}

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = DEFAULT_COLLECTION

    cmd = ["python", "/work/scripts/hybrid_search.py", "--limit", str(int(limit))]
    if per_path and int(per_path) > 0:
        cmd += ["--per-path", str(int(per_path))]
    for q in queries:
        cmd += ["--query", q]

    res = _run(cmd, env=env)

    # Parse hybrid_search tab-separated output: score\tpath\tsymbol_or_path\tstart-end
    results = []
    try:
        for line in (res.get("stdout") or "").splitlines():
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            score_s, path, symbol, range_s = parts
            try:
                start_s, end_s = range_s.split("-", 1)
                start_line = int(start_s)
                end_line = int(end_s)
            except Exception:
                start_line = 0
                end_line = 0
            try:
                score = float(score_s)
            except Exception:
                score = 0.0
            results.append({
                "score": score,
                "path": path,
                "symbol": symbol,
                "start_line": start_line,
                "end_line": end_line,
            })
    except Exception:
        pass

    return {
        "args": {"queries": queries, "limit": int(limit), "per_path": int(per_path), "collection": DEFAULT_COLLECTION},
        "results": results,
        **res,
    }



if __name__ == "__main__":
    transport = os.environ.get("FASTMCP_TRANSPORT", "sse").strip().lower()
    if transport == "stdio":
        # Run over stdio (for clients that don't support SSE)
        mcp.run(transport="stdio")
    else:
        # Serve over SSE at /sse on the configured host/port
        mcp.settings.host = HOST
        mcp.settings.port = PORT
        mcp.run(transport="sse")

