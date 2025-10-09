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
    """Index the mounted path (/work) or a subdirectory. Args: subdir?, recreate? (bool), collection?, repo_name?"""
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


if __name__ == "__main__":
    # Configure host/port then serve over SSE at /sse
    mcp.settings.host = HOST
    mcp.settings.port = PORT
    mcp.run(transport="sse")

