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
- QDRANT_URL (e.g., http://qdrant:6333) — server expects Qdrant reachable via this env
- COLLECTION_NAME (default: my-collection)

Conventions:
- Repo content must be mounted at /work inside containers
- Clients must not send null values for tool args; omit field or pass empty string ""
- To index repo root: use qdrant_index_root with no args, or qdrant_index with subdir=""

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
DEFAULT_REPO = "workspace"
MAX_LOG_TAIL = int(os.environ.get("MCP_MAX_LOG_TAIL", "4000"))

mcp = FastMCP(APP_NAME)


def _run(cmd: list[str], env: Optional[Dict[str, str]] = None, timeout: int = 60) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "code": -1,
            "stdout": (e.stdout or "")[-MAX_LOG_TAIL:] if e.stdout else "",
            "stderr": f"Command timed out after {timeout}s\n" + ((e.stderr or "")[-(MAX_LOG_TAIL-100):] if e.stderr else ""),
        }
    # Truncate to the last MAX_LOG_TAIL characters (tail-only) for both stdout and stderr
    def _cap_tail(s: str) -> str:
        if not s:
            return s
        return s if len(s) <= MAX_LOG_TAIL else s[-MAX_LOG_TAIL:]
    return {
        "ok": proc.returncode == 0,
        "code": proc.returncode,
        "stdout": _cap_tail(proc.stdout),
        "stderr": _cap_tail(proc.stderr),
    }


# Lightweight tokenizer and snippet highlighter
import re
_STOP = {"the","a","an","of","in","on","for","and","or","to","with","by","is","are","be","this","that"}

def _split_ident(s: str):
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]

def _tokens_from_queries(qs):
    toks = []
    for q in qs:
        toks.extend(_split_ident(q))
    seen = set(); out = []
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def _highlight_snippet(snippet: str, tokens: list[str]) -> str:
    if not snippet or not tokens:
        return snippet
    # longest first to avoid partial overlaps
    toks = sorted(set(tokens), key=len, reverse=True)
    def repl(m):
        return f"<<{m.group(0)}>>"
    for t in toks:
        try:
            pat = re.compile(re.escape(t), re.IGNORECASE)
            snippet = pat.sub(repl, snippet)
        except Exception:
            continue
    return snippet


@mcp.tool()
async def qdrant_index_root(recreate: Optional[bool] = None,
                            collection: Optional[str] = None) -> Dict[str, Any]:
    """Index the mounted root path (/work) with zero-arg safe defaults.
    Notes for IDE agents (Cursor/Windsurf/Augment):
    - Prefer this tool when you want to index the repo root without specifying params.
    - Do NOT send null values to tools; either omit a field or pass an empty string "".
    - Args:
      - recreate (bool, default false): drop and recreate collection schema if needed
      - collection (string, optional): defaults to env COLLECTION_NAME
    """
    coll = collection or DEFAULT_COLLECTION

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = ["python", "/work/scripts/ingest_code.py", "--root", "/work"]
    if recreate:
        cmd.append("--recreate")
    res = _run(cmd, env=env)
    return {"args": {"root": "/work", "collection": coll, "recreate": recreate}, **res}

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
async def qdrant_index(subdir: Optional[str] = None, recreate: Optional[bool] = None,
                 collection: Optional[str] = None) -> Dict[str, Any]:
    """Index the mounted path (/work) or a subdirectory.
    Important for IDE agents (Cursor/Windsurf/Augment):
    - Do NOT pass null values; omit a field or pass empty string "".
    - subdir: "" or omit to index repo root; or a relative path like "scripts"
    - recreate: bool (default false)
    - collection: string (optional; defaults to env COLLECTION_NAME)
    """
    root = "/work"
    if subdir:
        subdir = subdir.lstrip("/")
        root = os.path.join(root, subdir)
    coll = collection or DEFAULT_COLLECTION

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = [
        "python", "/work/scripts/ingest_code.py", "--root", root,
    ]
    if recreate:
        cmd.append("--recreate")
    res = _run(cmd, env=env)
    return {"args": {"root": root, "collection": coll, "recreate": recreate}, **res}


@mcp.tool()
async def qdrant_prune() -> Dict[str, Any]:
    """Prune stale points for the mounted path (/work)"""
    env = os.environ.copy()
    env["PRUNE_ROOT"] = "/work"
    cmd = ["python", "/work/scripts/prune.py"]
    res = _run(cmd, env=env)
    return res

@mcp.tool()
async def repo_search(
    query: Any = None,
    limit: Any = None,
    per_path: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    rerank_enabled: Any = None,
    rerank_top_n: Any = None,
    rerank_return_m: Any = None,
    rerank_timeout_ms: Any = None,
    highlight_snippet: Any = None,
    collection: Any = None,
) -> Dict[str, Any]:
    """Zero-config code search over the mounted repo via Qdrant using hybrid_search defaults.
    Args:
      - query: string or list of strings
      - limit: total number of results to return (default 10)
      - per_path: cap of results per file path to diversify output (default 2)
      - include_snippet/context_lines: embed code snippets near hit lines
      - rerank_*: optional ONNX reranker via rerank_local.py; graceful timeout fallback
      - highlight_snippet: emphasize matched tokens in snippet
    Notes:
      - No filters required; uses existing environment defaults (COLLECTION_NAME, QDRANT_URL).
      - Returns structured results parsed from hybrid_search JSONL output when possible.
    """
    # Leniency shim: coerce null/invalid args to sane defaults so buggy clients don't fail schema
    def _to_int(x, default):
        try:
            if x is None or (isinstance(x, str) and x.strip() == ""):
                return default
            return int(x)
        except Exception:
            return default
    def _to_bool(x, default):
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        if s in {"1","true","yes","on"}: return True
        if s in {"0","false","no","off"}: return False
        return default
    def _to_str(x, default=""):
        if x is None:
            return default
        return str(x)

    # Coerce incoming args (which may be null) to proper types
    limit = _to_int(limit, 10)
    per_path = _to_int(per_path, 2)
    include_snippet = _to_bool(include_snippet, False)
    context_lines = _to_int(context_lines, 2)
    rerank_enabled = _to_bool(rerank_enabled, False)
    rerank_top_n = _to_int(rerank_top_n, 50)
    rerank_return_m = _to_int(rerank_return_m, 12)
    rerank_timeout_ms = _to_int(rerank_timeout_ms, 120)
    highlight_snippet = _to_bool(highlight_snippet, True)
    collection = (_to_str(collection, "").strip() or DEFAULT_COLLECTION)

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
    env["COLLECTION_NAME"] = (collection or DEFAULT_COLLECTION)

    # Try hybrid search first (JSONL output) unless rerank path is requested exclusively
    cmd = ["python", "/work/scripts/hybrid_search.py", "--limit", str(int(limit)), "--json"]
    if per_path and int(per_path) > 0:
        cmd += ["--per-path", str(int(per_path))]
    for q in queries:
        cmd += ["--query", q]

    res = _run(cmd, env=env)

    results = []
    json_lines = []
    for line in (res.get("stdout") or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            json_lines.append(obj)
        except Exception:
            continue

    # Optional rerank fallback path: if enabled, attempt; on timeout or error, keep hybrid
    used_rerank = False
    if rerank_enabled:
        try:
            rq = queries[0] if queries else ""
            rcmd = [
                "python", "/work/scripts/rerank_local.py",
                "--query", rq,
                "--topk", str(int(rerank_top_n)),
                "--limit", str(int(rerank_return_m)),
            ]
            r = subprocess.run(rcmd, capture_output=True, text=True, timeout=max(0.1, int(rerank_timeout_ms)/1000.0), env=env)
            if r.returncode == 0 and r.stdout.strip():
                tmp = []
                for ln in r.stdout.splitlines():
                    parts = ln.strip().split("\t")
                    if len(parts) != 4:
                        continue
                    score_s, path, symbol, range_s = parts
                    try:
                        start_s, end_s = range_s.split("-", 1)
                        start_line = int(start_s); end_line = int(end_s)
                    except Exception:
                        start_line = 0; end_line = 0
                    try:
                        score = float(score_s)
                    except Exception:
                        score = 0.0
                    item = {"score": score, "path": path, "symbol": symbol, "start_line": start_line, "end_line": end_line, "why": [f"rerank_onnx:{score:.3f}"]}
                    tmp.append(item)
                if tmp:
                    results = tmp
                    used_rerank = True
        except subprocess.TimeoutExpired:
            used_rerank = False
        except Exception:
            used_rerank = False

    if not used_rerank:
        # Build results from hybrid JSON lines
        for obj in json_lines:
            item = {
                "score": float(obj.get("score", 0.0)),
                "path": obj.get("path", ""),
                "symbol": obj.get("symbol", ""),
                "start_line": int(obj.get("start_line") or 0),
                "end_line": int(obj.get("end_line") or 0),
                "why": obj.get("why", []),
                "components": obj.get("components", {}),
            }
            results.append(item)

    # Optionally add snippets (with highlighting)
    toks = _tokens_from_queries(queries)
    if include_snippet:
        for item in results:
            path = item.get("path")
            sl = int(item.get("start_line") or 0)
            el = int(item.get("end_line") or 0)
            if not path or not sl:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                si = max(1, sl - max(1, int(context_lines)))
                ei = min(len(lines), max(sl, el) + max(1, int(context_lines)))
                snippet = "".join(lines[si-1:ei])
                if highlight_snippet:
                    snippet = _highlight_snippet(snippet, toks)
                item["snippet"] = snippet
            except Exception:
                item["snippet"] = ""

    return {
        "args": {
            "queries": queries,
            "limit": int(limit),
            "per_path": int(per_path),
            "include_snippet": bool(include_snippet),
            "context_lines": int(context_lines),
            "rerank_enabled": bool(rerank_enabled),
            "rerank_top_n": int(rerank_top_n),
            "rerank_return_m": int(rerank_return_m),
            "rerank_timeout_ms": int(rerank_timeout_ms),
            "collection": (collection or DEFAULT_COLLECTION),
        },
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

