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
from typing import Any, Dict, Optional, List

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
async def qdrant_status(collection: Optional[str] = None, max_points: Optional[int] = None, batch: Optional[int] = None) -> Dict[str, Any]:
    """Report collection size and approximate last index times.
    Args:
      - collection: override collection name (defaults to env COLLECTION_NAME)
      - max_points: safety cap on points to scan when estimating last timestamps (default 5000)
      - batch: page size for scroll (default 1000)
    """
    coll = collection or DEFAULT_COLLECTION
    try:
        from qdrant_client import QdrantClient
        import datetime as _dt
        client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
        # Count points
        try:
            cnt_res = client.count(collection_name=coll, exact=True)
            total = int(getattr(cnt_res, "count", 0))
        except Exception:
            total = 0
        # Scan a limited number of points to estimate last timestamps
        max_points = int(max_points) if max_points not in (None, "") else int(os.environ.get("MCP_STATUS_MAX_POINTS", "5000"))
        batch = int(batch) if batch not in (None, "") else 1000
        scanned = 0
        last_ing = None
        last_mod = None
        next_page = None
        while scanned < max_points:
            limit = min(batch, max_points - scanned)
            try:
                pts, next_page = client.scroll(collection_name=coll, limit=limit, offset=next_page, with_payload=True, with_vectors=False)
            except Exception:
                # Fallback without offset keyword (older clients)
                pts, next_page = client.scroll(collection_name=coll, limit=limit, with_payload=True, with_vectors=False)
            if not pts:
                break
            scanned += len(pts)
            for p in pts:
                md = (p.payload or {}).get("metadata") or {}
                ti = md.get("ingested_at")
                tm = md.get("last_modified_at")
                if isinstance(ti, int):
                    last_ing = ti if last_ing is None else max(last_ing, ti)
                if isinstance(tm, int):
                    last_mod = tm if last_mod is None else max(last_mod, tm)
            if not next_page:
                break
        def _iso(ts):
            if isinstance(ts, int) and ts > 0:
                try:
                    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat()
                except Exception:
                    return ""
            return ""
        return {
            "collection": coll,
            "count": total,
            "scanned_points": scanned,
            "last_ingested_at": {"unix": last_ing or 0, "iso": _iso(last_ing)},
            "last_modified_at": {"unix": last_mod or 0, "iso": _iso(last_mod)},
        }
    except Exception as e:
        return {"collection": coll, "error": str(e)}



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
    # Structured filters (optional; mirrors hybrid_search flags)
    language: Any = None,
    under: Any = None,
    kind: Any = None,
    symbol: Any = None,
    # Additional structured parity
    path_regex: Any = None,
    path_glob: Any = None,
    not_glob: Any = None,
    ext: Any = None,
    not_: Any = None,
    case: Any = None,
    # Response shaping
    compact: Any = None,
) -> Dict[str, Any]:
    """Zero-config code search over the mounted repo via Qdrant using hybrid_search defaults.
    Args:
      - query: string or list of strings
      - limit: total number of results to return (default 10)
      - per_path: cap of results per file path to diversify output (default 2)
      - include_snippet/context_lines: embed code snippets near hit lines
      - rerank_*: optional ONNX reranker via rerank_local.py; graceful timeout fallback
      - highlight_snippet: emphasize matched tokens in snippet
      - collection: override target collection (default env COLLECTION_NAME)
      - language/under/kind/symbol: structured search filters (alternative to DSL tokens)
    Notes:
      - No filters required; uses existing environment defaults (COLLECTION_NAME, QDRANT_URL).
      - You can also pass DSL tokens inside the query text, e.g. "lang:python file:scripts/".
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

    language = _to_str(language, "").strip()
    under = _to_str(under, "").strip()
    kind = _to_str(kind, "").strip()
    symbol = _to_str(symbol, "").strip()
    path_regex = _to_str(path_regex, "").strip()
    # Normalize globs to lists (accept string or list)
    def _to_str_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            out = []
            for e in x:
                s = str(e).strip()
                if s:
                    out.append(s)
            return out
        s = str(x).strip()
        if not s:
            return []
        # support comma-separated shorthand
        return [t.strip() for t in s.split(",") if t.strip()]
    path_globs = _to_str_list(path_glob)
    not_globs = _to_str_list(not_glob)
    ext = _to_str(ext, "").strip()
    not_ = _to_str(not_, "").strip()
    case = _to_str(case, "").strip()
    compact_raw = compact
    compact = _to_bool(compact, False)

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
    if language:
        cmd += ["--language", language]
    if under:
        cmd += ["--under", under]
    if kind:
        cmd += ["--kind", kind]
    if symbol:
        cmd += ["--symbol", symbol]
    if ext:
        cmd += ["--ext", ext]
    if not_:
        cmd += ["--not", not_]
    if case:
        cmd += ["--case", case]
    if path_regex:
        cmd += ["--path-regex", path_regex]
    for g in path_globs:
        cmd += ["--path-glob", g]
    for g in not_globs:
        cmd += ["--not-glob", g]
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

    # Smart default: compact true for multi-query calls if compact not explicitly set
    if (len(queries) > 1) and (compact_raw is None or (isinstance(compact_raw, str) and compact_raw.strip() == "")):
        compact = True

    # Compact mode: return only path and line range
    if compact:
        results = [
            {
                "path": r.get("path", ""),
                "start_line": int(r.get("start_line") or 0),
                "end_line": int(r.get("end_line") or 0),
            }
            for r in results
        ]

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
            "language": language,
            "under": under,
            "kind": kind,
            "symbol": symbol,
            "ext": ext,
            "not": not_,
            "case": case,
            "path_regex": path_regex,
            "path_glob": path_globs,
            "not_glob": not_globs,
            "compact": bool(compact),
        },
        "total": len(results),
        "results": results,
        **res,
    }



@mcp.tool()
async def context_search(
    # Core query + limits
    query: Any = None,
    limit: Any = None,
    per_path: Any = None,
    # Include memory hits and blending controls
    include_memories: Any = None,
    memory_weight: Any = None,
    per_source_limits: Any = None,  # e.g., {"code": 5, "memory": 3}
    # Pass-through structured filters (same as repo_search)
    include_snippet: Any = None,
    context_lines: Any = None,
    rerank_enabled: Any = None,
    rerank_top_n: Any = None,
    rerank_return_m: Any = None,
    rerank_timeout_ms: Any = None,
    highlight_snippet: Any = None,
    collection: Any = None,
    language: Any = None,
    under: Any = None,
    kind: Any = None,
    symbol: Any = None,
    path_regex: Any = None,
    path_glob: Any = None,
    not_glob: Any = None,
    ext: Any = None,
    not_: Any = None,
    case: Any = None,
    compact: Any = None,
) -> Dict[str, Any]:
    """Context-aware search that optionally blends code hits with memory hits.

    - Applies memory-derived defaults (safe subset) automatically:
      * compact=true if multi-query and compact not explicitly provided
      * per_path=1 if not explicitly provided
    - When include_memories is true, queries Qdrant directly for memory-like points
      (payloads lacking code path metadata) and blends them with code results.
    - memory_weight scales memory scores when merging.
    """
    # Normalize inputs
    coll = (collection or DEFAULT_COLLECTION) or ""
    try:
        lim = int(limit) if (limit is not None and str(limit).strip() != "") else 10
    except Exception:
        lim = 10
    try:
        per_path_val = int(per_path) if (per_path is not None and str(per_path).strip() != "") else 1
    except Exception:
        per_path_val = 1

    # Normalize queries to list
    queries: List[str] = []
    if isinstance(query, (list, tuple)):
        queries = [str(q) for q in query]
    elif query is not None and str(query).strip() != "":
        queries = [str(query)]

    # Smart defaults inspired by stored preferences, but without external calls
    compact_raw = compact
    smart_compact = False
    if len(queries) > 1 and (compact_raw is None or (isinstance(compact_raw, str) and compact_raw.strip() == "")):
        smart_compact = True
    eff_compact = True if (smart_compact or (str(compact_raw).lower() == "true")) else False

    # Per-source limits
    code_limit = lim
    mem_limit = 0
    include_mem = False
    if include_memories is not None and str(include_memories).lower() in ("true", "1", "yes"):  # opt-in
        include_mem = True
        # Parse per_source_limits if provided
        code_limit = lim
        mem_limit = min(3, lim)  # sensible default
        try:
            if isinstance(per_source_limits, dict):
                code_limit = int(per_source_limits.get("code", code_limit))
                mem_limit = int(per_source_limits.get("memory", mem_limit))
        except Exception:
            pass

    # First: run code search via internal repo_search for consistent behavior
    code_res = await repo_search(
        query=queries if len(queries) > 1 else (queries[0] if queries else ""),
        limit=code_limit,
        per_path=per_path_val,
        include_snippet=include_snippet,
        context_lines=context_lines,
        rerank_enabled=rerank_enabled,
        rerank_top_n=rerank_top_n,
        rerank_return_m=rerank_return_m,
        rerank_timeout_ms=rerank_timeout_ms,
        highlight_snippet=highlight_snippet,
        collection=coll,
        language=language,
        under=under,
        kind=kind,
        symbol=symbol,
        path_regex=path_regex,
        path_glob=path_glob,
        not_glob=not_glob,
        ext=ext,
        not_=not_,
        case=case,
        compact=eff_compact,
    )

    # Shape code results to a common schema
    code_hits: List[Dict[str, Any]] = []
    if isinstance(code_res, dict):
        items = code_res.get("results") or code_res.get("data") or code_res.get("items")
        # If compact mode was used, results may be a list; support both shapes
        items = items if items is not None else code_res.get("results", code_res)
    else:
        items = code_res
    # Normalize list
    if isinstance(items, list):
        for r in items:
            if isinstance(r, dict):
                ch = {
                    "source": "code",
                    "score": float(r.get("score") or r.get("s") or 0.0),
                    "path": r.get("path"),
                    "symbol": r.get("symbol", ""),
                    "start_line": r.get("start_line"),
                    "end_line": r.get("end_line"),
                    "_raw": r,
                }
                code_hits.append(ch)

    # Optionally: fetch memory hits directly from Qdrant
    mem_hits: List[Dict[str, Any]] = []
    if include_mem and mem_limit > 0 and queries:
        try:
            from qdrant_client import QdrantClient, models  # type: ignore
            from fastembed import TextEmbedding  # type: ignore
            from scripts.utils import sanitize_vector_name  # local util

            client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            vec_name = sanitize_vector_name(model_name)
            model = TextEmbedding(model_name=model_name)

            # Use first query for memory retrieval (can be extended to multi-query fusion)
            qtext = " ".join([q for q in queries if q]).strip() or queries[0]
            v = next(model.embed([qtext])).tolist()
            # Fetch a few extra and filter client-side for "memory-like" payloads
            k = max(mem_limit, 5)
            res = client.search(
                collection_name=coll,
                query_vector={"name": vec_name, "vector": v},
                limit=k,
                with_payload=True,
            )
            for pt in res:
                payload = (getattr(pt, "payload", {}) or {})
                md = payload.get("metadata") or {}
                # Heuristic: memory entries typically lack code path/lines
                path = str(md.get("path") or "")
                start_line = md.get("start_line")
                end_line = md.get("end_line")
                content = payload.get("content") or payload.get("text") or payload.get("information") or md.get("information")
                is_memory_like = (not path) or (start_line in (None, 0) and end_line in (None, 0))
                if is_memory_like and content:
                    mem_hits.append({
                        "source": "memory",
                        "score": float(getattr(pt, "score", 0.0) or 0.0),
                        "content": content,
                        "metadata": md,
                    })
        except Exception:  # pragma: no cover
            pass

    # Fallback: lightweight substring scan over a capped scroll if vector name mismatch
    if include_mem and mem_limit > 0 and not mem_hits and queries:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
            terms = [str(t).lower() for t in queries if t]
            checked = 0
            cap = 2000
            page = None
            while len(mem_hits) < mem_limit and checked < cap:
                sc, page = client.scroll(
                    collection_name=coll,
                    with_payload=True,
                    with_vectors=False,
                    limit=500,
                    offset=page,
                )
                if not sc:
                    break
                for pt in sc:
                    payload = (getattr(pt, "payload", {}) or {})
                    md = payload.get("metadata") or {}
                    path = str(md.get("path") or "")
                    start_line = md.get("start_line")
                    end_line = md.get("end_line")
                    content = payload.get("content") or payload.get("text") or payload.get("information") or md.get("information")
                    is_memory_like = (not path) or (start_line in (None, 0) and end_line in (None, 0))
                    if not (is_memory_like and content):
                        continue
                    low = str(content).lower()
                    if any(t in low for t in terms):
                        mem_hits.append({
                            "source": "memory",
                            "score": 0.5,  # nominal score for substring match; blended via memory_weight
                            "content": content,
                            "metadata": md,
                        })
                        if len(mem_hits) >= mem_limit:
                            break
                checked += len(sc)
        except Exception:
            pass


    # Blend results
    try:
        mw = float(memory_weight) if (memory_weight is not None and str(memory_weight).strip() != "") else 0.3
    except Exception:
        mw = 0.3

    blended: List[Dict[str, Any]] = []
    for h in code_hits:
        blended.append({**h, "score": float(h.get("score", 0.0))})
    for h in mem_hits:
        blended.append({**h, "score": float(h.get("score", 0.0)) * mw})

    # Sort by score descending and truncate to limit
    blended.sort(key=lambda x: (-float(x.get("score", 0.0)), x.get("source", ""), str(x.get("path", ""))))
    blended = blended[:lim]

    # Compact shaping if requested
    if eff_compact:
        compacted: List[Dict[str, Any]] = []
        for b in blended:
            if b.get("source") == "code":
                compacted.append({
                    "source": "code",
                    "path": b.get("path"),
                    "start_line": b.get("start_line") or 0,
                    "end_line": b.get("end_line") or 0,
                })
            else:
                compacted.append({
                    "source": "memory",
                    "content": (b.get("content") or "")[:500],
                })
        return {"results": compacted, "total": len(compacted)}

    return {"results": blended, "total": len(blended)}

@mcp.tool()
async def code_search(
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
    language: Any = None,
    under: Any = None,
    kind: Any = None,
    symbol: Any = None,
    path_regex: Any = None,
    path_glob: Any = None,
    not_glob: Any = None,
    ext: Any = None,
    not_: Any = None,
    case: Any = None,
    compact: Any = None,
) -> Dict[str, Any]:
    """Alias of repo_search with the same arguments; provided for better discoverability."""
    return await repo_search(
        query=query,
        limit=limit,
        per_path=per_path,
        include_snippet=include_snippet,
        context_lines=context_lines,
        rerank_enabled=rerank_enabled,
        rerank_top_n=rerank_top_n,
        rerank_return_m=rerank_return_m,
        rerank_timeout_ms=rerank_timeout_ms,
        highlight_snippet=highlight_snippet,
        collection=collection,
        language=language,
        under=under,
        kind=kind,
        symbol=symbol,
        path_regex=path_regex,
        path_glob=path_glob,
        not_glob=not_glob,
        ext=ext,
        not_=not_,
        case=case,
        compact=compact,
    )



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

