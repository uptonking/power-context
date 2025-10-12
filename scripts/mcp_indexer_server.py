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
import asyncio
import uuid

import os
import subprocess
import threading
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
MAX_LOG_TAIL = int(os.environ.get("MCP_MAX_LOG_TAIL", "4000"))
SNIPPET_MAX_BYTES = int(os.environ.get("MCP_SNIPPET_MAX_BYTES", "8192") or 8192)

mcp = FastMCP(APP_NAME)



# Async subprocess runner to avoid blocking event loop
async def _run_async(cmd: list[str], env: Optional[Dict[str, str]] = None, timeout: int = 60) -> Dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            code = proc.returncode
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return {
                "ok": False,
                "code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
            }
        stdout = (stdout_b or b"").decode("utf-8", errors="ignore")
        stderr = (stderr_b or b"").decode("utf-8", errors="ignore")
        def _cap_tail(s: str) -> str:
            if not s:
                return s
            return s if len(s) <= MAX_LOG_TAIL else ("...[tail truncated]\n" + s[-MAX_LOG_TAIL:])
        return {"ok": code == 0, "code": code, "stdout": _cap_tail(stdout), "stderr": _cap_tail(stderr)}
    except Exception as e:
        return {"ok": False, "code": -2, "stdout": "", "stderr": str(e)}

# Embedding model cache to avoid re-initialization costs
_EMBED_MODEL_CACHE: Dict[str, Any] = {}
_EMBED_MODEL_LOCKS: Dict[str, threading.Lock] = {}

def _get_embedding_model(model_name: str):
    try:
        from fastembed import TextEmbedding  # type: ignore
    except Exception:
        raise
    m = _EMBED_MODEL_CACHE.get(model_name)
    if m is None:
        # Double-checked locking to avoid duplicate inits under concurrency
        lock = _EMBED_MODEL_LOCKS.setdefault(model_name, threading.Lock())
        with lock:
            m = _EMBED_MODEL_CACHE.get(model_name)
            if m is None:
                m = TextEmbedding(model_name=model_name)
                _EMBED_MODEL_CACHE[model_name] = m
    return m

# Lenient argument normalization to tolerate buggy clients (e.g., JSON-in-kwargs, booleans where strings expected)
from typing import Any as _Any, Dict as _Dict

def _maybe_parse_jsonish(obj: _Any):
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, str):
        return None
    s = obj.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return json.loads("{" + s + "}")
    except Exception:
        return None

def _extract_kwargs_payload(kwargs: _Any) -> _Dict[str, _Any]:
    try:
        if isinstance(kwargs, dict) and "kwargs" in kwargs:
            inner = kwargs.get("kwargs")
            if isinstance(inner, dict):
                return inner
            parsed = _maybe_parse_jsonish(inner)
            return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
    return {}

def _looks_jsonish_string(s: _Any) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    if t.startswith("{") and ":" in t:
        return True
    if t.endswith("}"):
        return True
    # quick heuristics for comma/colon pairs often seen when args are concatenated
    return ("," in t and ":" in t) or ('":' in t)

def _coerce_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default

def _coerce_int(x, default=0):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(x)
    except Exception:
        return default

def _coerce_str(x, default=""):
    if x is None:
        return default
    return str(x)

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
    # Leniency: if clients embed JSON in 'collection' (and include 'recreate'), parse it
    try:
        if _looks_jsonish_string(collection):
            _parsed = _maybe_parse_jsonish(collection)
            if isinstance(_parsed, dict):
                collection = _parsed.get("collection", collection)
                if recreate is None and "recreate" in _parsed:
                    recreate = _coerce_bool(_parsed.get("recreate"), False)
    except Exception:
        pass

    coll = collection or DEFAULT_COLLECTION

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = ["python", "/work/scripts/ingest_code.py", "--root", "/work"]
    if recreate:
        cmd.append("--recreate")
    res = await _run_async(cmd, env=env)
    return {"args": {"root": "/work", "collection": coll, "recreate": recreate}, **res}

@mcp.tool()
async def qdrant_list(**kwargs) -> Dict[str, Any]:
    """List Qdrant collections (ignores any extra params)"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
        cols_info = await asyncio.to_thread(client.get_collections)
        return {"collections": [c.name for c in cols_info.collections]}
    except ImportError:
        return {"error": "qdrant_client is not installed in this container"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def memory_store(information: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       collection: Optional[str] = None) -> Dict[str, Any]:
    """Store a memory-like entry directly into Qdrant using the default collection.
    - information: free-form text to remember
    - metadata: optional tags (e.g., {"kind":"preference","source":"memory"})
    - collection: override target collection (defaults to env COLLECTION_NAME)
    """
    try:
        from qdrant_client import QdrantClient, models  # type: ignore
        from fastembed import TextEmbedding  # type: ignore
        import time, hashlib, re, math
        from scripts.utils import sanitize_vector_name
        from scripts.ingest_code import ensure_collection as _ensure_collection  # type: ignore
    except Exception as e:  # pragma: no cover
        return {"error": f"deps: {e}"}

    if not information or not str(information).strip():
        return {"error": "information is required"}

    coll = (collection or DEFAULT_COLLECTION) or ""
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vector_name = sanitize_vector_name(model_name)

    # Minimal lexical hashing (aligns with ingest_code defaults)
    LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
    LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)

    def _split_ident_lex(s: str):
        parts = re.split(r"[^A-Za-z0-9]+", s)
        out: list[str] = []
        for p in parts:
            if not p:
                continue
            segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
            out.extend([x for x in segs if x])
        return [x.lower() for x in out if x]

    def _lex_hash_vector(text: str, dim: int = LEX_VECTOR_DIM) -> list[float]:
        if not text:
            return [0.0] * dim
        vec = [0.0] * dim
        toks = _split_ident_lex(text)
        if not toks:
            return vec
        for t in toks:
            h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
        # L2 normalize to align with ingest_code._lex_hash_vector
        norm = (sum(v*v for v in vec) or 0.0) ** 0.5 or 1.0
        return [v / norm for v in vec]

    # Build vectors (cached embedding model)
    model = _get_embedding_model(model_name)
    dense = next(model.embed([str(information)])).tolist()

    lex = _lex_hash_vector(str(information))

    # Upsert
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
        # Ensure collection and named vectors exist (dense + lexical)
        try:
            await asyncio.to_thread(lambda: _ensure_collection(client, coll, len(dense), vector_name))
        except Exception:
            pass
        pid = str(uuid.uuid4())
        payload = {"information": str(information), "metadata": metadata or {"kind": "memory", "source": "memory"}}
        point = models.PointStruct(id=pid, vector={vector_name: dense, LEX_VECTOR_NAME: lex}, payload=payload)
        await asyncio.to_thread(lambda: client.upsert(collection_name=coll, points=[point], wait=True))
        return {"ok": True, "id": pid, "collection": coll, "vector_name": vector_name}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def qdrant_status(collection: Optional[str] = None, max_points: Optional[int] = None, batch: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """Report collection size and approximate last index times. Extra params are ignored.
    Args:
      - collection: override collection name (defaults to env COLLECTION_NAME)
      - max_points: safety cap on points to scan when estimating last timestamps (default 5000)
      - batch: page size for scroll (default 1000)
    """
    # Leniency: absorb 'kwargs' JSON payload some clients send instead of top-level args
    try:
        _extra = _extract_kwargs_payload(kwargs)
        if _extra and not collection:
            collection = _extra.get("collection", collection)
        if _extra and max_points in (None, "") and _extra.get("max_points") is not None:
            max_points = _coerce_int(_extra.get("max_points"), None)
        if _extra and batch in (None, "") and _extra.get("batch") is not None:
            batch = _coerce_int(_extra.get("batch"), None)
    except Exception:
        pass
    coll = collection or DEFAULT_COLLECTION
    try:
        from qdrant_client import QdrantClient
        import datetime as _dt
        client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
        # Count points
        try:
            cnt_res = await asyncio.to_thread(lambda: client.count(collection_name=coll, exact=True))
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
                pts, next_page = await asyncio.to_thread(lambda: client.scroll(collection_name=coll, limit=limit, offset=next_page, with_payload=True, with_vectors=False))
            except Exception:
                # Fallback without offset keyword (older clients)
                pts, next_page = await asyncio.to_thread(lambda: client.scroll(collection_name=coll, limit=limit, with_payload=True, with_vectors=False))
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
    # Leniency: parse JSON-ish payloads mistakenly sent in 'collection' or 'subdir'
    try:
        if _looks_jsonish_string(collection):
            _parsed = _maybe_parse_jsonish(collection)
            if isinstance(_parsed, dict):
                subdir = _parsed.get("subdir", subdir)
                collection = _parsed.get("collection", collection)
                if recreate is None and "recreate" in _parsed:
                    recreate = _coerce_bool(_parsed.get("recreate"), False)
        if _looks_jsonish_string(subdir):
            _parsed2 = _maybe_parse_jsonish(subdir)
            if isinstance(_parsed2, dict):
                subdir = _parsed2.get("subdir", subdir)
                collection = _parsed2.get("collection", collection)
                if recreate is None and "recreate" in _parsed2:
                    recreate = _coerce_bool(_parsed2.get("recreate"), False)
    except Exception:
        pass

    root = "/work"
    if subdir:
        subdir = subdir.lstrip("/")
        root = os.path.join(root, subdir)
    # Enforce /work sandbox
    real_root = os.path.realpath(root)
    if not (real_root == "/work" or real_root.startswith("/work/")):
        return {"ok": False, "error": "subdir escapes /work sandbox"}
    root = real_root
    coll = collection or DEFAULT_COLLECTION

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = [
        "python", "/work/scripts/ingest_code.py", "--root", root,
    ]
    if recreate:
        cmd.append("--recreate")
    res = await _run_async(cmd, env=env)
    return {"args": {"root": root, "collection": coll, "recreate": recreate}, **res}


@mcp.tool()
async def qdrant_prune(**kwargs) -> Dict[str, Any]:
    """Prune stale points for the mounted path (/work). Extra params are ignored."""
    env = os.environ.copy()
    env["PRUNE_ROOT"] = "/work"

    cmd = ["python", "/work/scripts/prune.py"]
    res = await _run_async(cmd, env=env)
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
    **kwargs,
) -> Dict[str, Any]:
    """Zero-config code search over the mounted repo via Qdrant hybrid search.

    Args:
      - query: string or list of strings
      - limit: total results (default 10)
      - per_path: max results per file (default 2)
      - include_snippet/context_lines: include snippet near hit lines
      - rerank_*: optional ONNX reranker; timeouts fall back to hybrid
      - collection: override target collection (default env COLLECTION_NAME)
      - language/under/kind/symbol/path_regex/path_glob/not_glob/ext/not_/case: optional filters
      - compact: if true, return only path and line range

    Returns:
      - {"results": [...], "total": N}
    """
    # Accept common alias keys from clients (top-level)
    try:
        if (limit is None or (isinstance(limit, str) and str(limit).strip() == "")) and ("top_k" in kwargs):
            limit = kwargs.get("top_k")
        if (query is None or (isinstance(query, str) and str(query).strip() == "")):
            q_alt = kwargs.get("q") or kwargs.get("text")
            if q_alt is not None:
                query = q_alt
    except Exception:
        pass

    # Leniency: absorb nested 'kwargs' JSON payload some clients send
    try:
        _extra = _extract_kwargs_payload(kwargs)
        if _extra:
            if (query is None or (isinstance(query, str) and query.strip() == "")):
                query = _extra.get("query") or _extra.get("queries")
            if limit in (None, "") and _extra.get("limit") is not None:
                limit = _extra.get("limit")
            if per_path in (None, "") and _extra.get("per_path") is not None:
                per_path = _extra.get("per_path")
            if include_snippet in (None, "") and _extra.get("include_snippet") is not None:
                include_snippet = _extra.get("include_snippet")
            if context_lines in (None, "") and _extra.get("context_lines") is not None:
                context_lines = _extra.get("context_lines")
            if rerank_enabled in (None, "") and _extra.get("rerank_enabled") is not None:
                rerank_enabled = _extra.get("rerank_enabled")
            if rerank_top_n in (None, "") and _extra.get("rerank_top_n") is not None:
                rerank_top_n = _extra.get("rerank_top_n")
            if rerank_return_m in (None, "") and _extra.get("rerank_return_m") is not None:
                rerank_return_m = _extra.get("rerank_return_m")
            if rerank_timeout_ms in (None, "") and _extra.get("rerank_timeout_ms") is not None:
                rerank_timeout_ms = _extra.get("rerank_timeout_ms")
            if highlight_snippet in (None, "") and _extra.get("highlight_snippet") is not None:
                highlight_snippet = _extra.get("highlight_snippet")
            if (collection is None or (isinstance(collection, str) and collection.strip() == "")) and _extra.get("collection"):
                collection = _extra.get("collection")
            if (language is None or (isinstance(language, str) and language.strip() == "")) and _extra.get("language"):
                language = _extra.get("language")
            if (under is None or (isinstance(under, str) and under.strip() == "")) and _extra.get("under"):
                under = _extra.get("under")
            if (kind is None or (isinstance(kind, str) and kind.strip() == "")) and _extra.get("kind"):
                kind = _extra.get("kind")
            if (symbol is None or (isinstance(symbol, str) and symbol.strip() == "")) and _extra.get("symbol"):
                symbol = _extra.get("symbol")
            if (path_regex is None or (isinstance(path_regex, str) and path_regex.strip() == "")) and _extra.get("path_regex"):
                path_regex = _extra.get("path_regex")
            if path_glob in (None, "") and _extra.get("path_glob") is not None:
                path_glob = _extra.get("path_glob")
            if not_glob in (None, "") and _extra.get("not_glob") is not None:
                not_glob = _extra.get("not_glob")
            if (ext is None or (isinstance(ext, str) and ext.strip() == "")) and _extra.get("ext"):
                ext = _extra.get("ext")
            if (not_ is None or (isinstance(not_, str) and not_.strip() == "")) and (_extra.get("not") or _extra.get("not_")):
                not_ = _extra.get("not") or _extra.get("not_")
            if (case is None or (isinstance(case, str) and case.strip() == "")) and _extra.get("case"):
                case = _extra.get("case")
            if compact in (None, "") and _extra.get("compact") is not None:
                compact = _extra.get("compact")
    except Exception:
        pass

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
    # Reranker: allow env-defaults to enable without client args
    rerank_env_default = str(os.environ.get("RERANKER_ENABLED", "")).strip().lower() in {"1","true","yes","on"}
    rerank_enabled = _to_bool(rerank_enabled, rerank_env_default)
    rerank_top_n = _to_int(rerank_top_n, int(os.environ.get("RERANKER_TOPN", "50") or 50))
    rerank_return_m = _to_int(rerank_return_m, int(os.environ.get("RERANKER_RETURN_M", "12") or 12))
    rerank_timeout_ms = _to_int(rerank_timeout_ms, int(os.environ.get("RERANKER_TIMEOUT_MS", "120") or 120))
    highlight_snippet = _to_bool(highlight_snippet, True)
    collection = (_to_str(collection, "").strip() or os.environ.get("COLLECTION_NAME", DEFAULT_COLLECTION))

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

    # Accept top-level alias `queries` as a drop-in for `query`
    # Many clients send queries=[...] instead of query=[...]
    if "queries" in kwargs and kwargs.get("queries") is not None:
        query = kwargs.get("queries")

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
    env["COLLECTION_NAME"] = collection

    results = []
    json_lines = []

    # In-process hybrid search (optional)

    # Default subprocess result placeholder (for consistent response shape)
    res = {"ok": True, "code": 0, "stdout": "", "stderr": ""}

    use_hybrid_inproc = str(os.environ.get("HYBRID_IN_PROCESS", "")).strip().lower() in {"1","true","yes","on"}
    if use_hybrid_inproc:
        try:
            from scripts.hybrid_search import run_hybrid_search  # type: ignore
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            model = _get_embedding_model(model_name)
            # In-process path_glob/not_glob accept a single string; reduce list inputs safely
            items = run_hybrid_search(
                queries=queries,
                limit=int(limit),
                per_path=int(per_path) if per_path else 1,
                language=language or None,
                under=under or None,
                kind=kind or None,
                symbol=symbol or None,
                ext=ext or None,
                not_filter=not_ or None,
                case=case or None,
                path_regex=path_regex or None,
                path_glob=(path_globs or None),
                not_glob=(not_globs or None),
                expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower() in {"1","true","yes","on"},
                model=model,
            )
            # items are already in structured dict form
            json_lines = items  # reuse downstream shaping
        except Exception as e:
            # Fallback to subprocess path if in-process fails
            use_hybrid_inproc = False

    if not use_hybrid_inproc:
        # Try hybrid search via subprocess (JSONL output)
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

        res = await _run_async(cmd, env=env)
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
        # Prefer fusion-aware reranking over hybrid candidates when available
        try:
            if json_lines:
                from scripts.rerank_local import rerank_local as _rr_local  # type: ignore
                import concurrent.futures as _fut
                rq = queries[0] if queries else ""
                # Prepare candidate docs from top-N hybrid hits (path+symbol + small snippet)
                cand_objs = list(json_lines[: int(rerank_top_n)])
                def _doc_for(obj: dict) -> str:
                    path = str(obj.get("path") or "")
                    symbol = str(obj.get("symbol") or "")
                    header = f"{symbol} — {path}".strip()
                    sl = int(obj.get("start_line") or 0)
                    el = int(obj.get("end_line") or 0)
                    if not path or not sl:
                        return header
                    try:
                        p = path
                        if not os.path.isabs(p):
                            p = os.path.join("/work", p)
                        realp = os.path.realpath(p)
                        if not (realp == "/work" or realp.startswith("/work/")):
                            return header
                        with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                        ctx = max(1, int(context_lines)) if 'context_lines' in locals() else 2
                        si = max(1, sl - ctx)
                        ei = min(len(lines), max(sl, el) + ctx)
                        snippet = "".join(lines[si-1:ei]).strip()
                        return (header + ("\n" + snippet if snippet else "")).strip()
                    except Exception:
                        return header
                # Build docs concurrently
                max_workers = min(8, (os.cpu_count() or 4) * 2)
                with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    docs = list(ex.map(_doc_for, cand_objs))
                pairs = [(rq, d) for d in docs]
                scores = _rr_local(pairs)
                ranked = sorted(zip(scores, cand_objs), key=lambda x: x[0], reverse=True)
                tmp = []
                for s, obj in ranked[: int(rerank_return_m)]:
                    item = {
                        "score": float(s),
                        "path": obj.get("path", ""),
                        "symbol": obj.get("symbol", ""),
                        "start_line": int(obj.get("start_line") or 0),
                        "end_line": int(obj.get("end_line") or 0),
                        "why": obj.get("why", []) + [f"rerank_onnx:{float(s):.3f}"],
                        "components": (obj.get("components") or {}) | {"rerank_onnx": float(s)},
                    }
                    tmp.append(item)
                if tmp:
                    results = tmp
                    used_rerank = True
        except Exception:
            used_rerank = False
        # Fallback paths (in-process reranker dense candidates, then subprocess)
        if not used_rerank:
            use_rerank_inproc = str(os.environ.get("RERANK_IN_PROCESS", "")).strip().lower() in {"1","true","yes","on"}
            if use_rerank_inproc:
                try:
                    from scripts.rerank_local import rerank_in_process  # type: ignore
                    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                    model = _get_embedding_model(model_name)
                    rq = queries[0] if queries else ""
                    items = rerank_in_process(
                        query=rq,
                        topk=int(rerank_top_n),
                        limit=int(rerank_return_m),
                        language=language or None,
                        under=under or None,
                        model=model,
                    )
                    if items:
                        results = items
                        used_rerank = True
                except Exception:
                    use_rerank_inproc = False
            if not use_rerank_inproc and not used_rerank:
                try:
                    rq = queries[0] if queries else ""
                    rcmd = [
                        "python", "/work/scripts/rerank_local.py",
                        "--query", rq,
                        "--topk", str(int(rerank_top_n)),
                        "--limit", str(int(rerank_return_m)),
                    ]
                    if language:
                        rcmd += ["--language", language]
                    if under:
                        rcmd += ["--under", under]
                    if os.environ.get("MCP_DEBUG_RERANK", "").strip():
                        try:
                            print("RERANK_CMD:", " ".join(rcmd))
                        except Exception:
                            pass
                    _floor_ms = int(os.environ.get("RERANK_TIMEOUT_FLOOR_MS", "1000"))
                    try:
                        _req_ms = int(rerank_timeout_ms)
                    except Exception:
                        _req_ms = _floor_ms
                    _eff_ms = max(_floor_ms, _req_ms)
                    _t_sec = max(0.1, _eff_ms / 1000.0)
                    rres = await _run_async(rcmd, env=env, timeout=_t_sec)
                    if os.environ.get("MCP_DEBUG_RERANK", "").strip():
                        try:
                            print("RERANK_RET:", rres.get("code"), "OUT_LEN:", len((rres.get("stdout") or "").strip()), "ERR_TAIL:", (rres.get("stderr") or "")[ -200: ])
                        except Exception:
                            pass
                    if rres.get("ok") and (rres.get("stdout") or "").strip():
                        tmp = []
                        for ln in (rres.get("stdout") or "").splitlines():
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
                # Enforce /work sandbox for snippet path
                raw_path = str(path)
                p = raw_path
                if not os.path.isabs(p):
                    p = os.path.join("/work", p)
                realp = os.path.realpath(p)
                if not (realp == "/work" or realp.startswith("/work/")):
                    item["snippet"] = ""
                    continue
                with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                si = max(1, sl - max(1, int(context_lines)))
                ei = min(len(lines), max(sl, el) + max(1, int(context_lines)))
                snippet = "".join(lines[si-1:ei])
                if highlight_snippet:
                    snippet = _highlight_snippet(snippet, toks)
                # Enforce strict size cap after highlighting
                if len(snippet.encode("utf-8", "ignore")) > SNIPPET_MAX_BYTES:
                    _suffix = "\n...[snippet truncated]"
                    _sb = _suffix.encode("utf-8")
                    _bytes = snippet.encode("utf-8", "ignore")
                    _keep = max(0, SNIPPET_MAX_BYTES - len(_sb))
                    _trimmed = _bytes[:_keep]
                    snippet = _trimmed.decode("utf-8", "ignore") + _suffix
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
            "collection": collection,
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
        "used_rerank": bool(used_rerank),
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
    **kwargs,
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
    mcoll = (os.environ.get("MEMORY_COLLECTION_NAME") or coll) or ""
    use_sse_memory = str(os.environ.get("MEMORY_SSE_ENABLED", "false")).lower() in ("1","true","yes")
    # Auto-detect memory collection if not explicitly set
    if include_memories and not os.environ.get("MEMORY_COLLECTION_NAME"):
        try:
            from qdrant_client import QdrantClient  # type: ignore
            client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
            info = await asyncio.to_thread(client.get_collections)
            best_name = None
            best_hits = -1
            for c in info.collections:
                name = getattr(c, "name", None)
                if not name:
                    continue
                # Sample a small page for memory-like payloads
                try:
                    pts, _ = await asyncio.to_thread(lambda: client.scroll(collection_name=name, with_payload=True, with_vectors=False, limit=300))
                    hits = 0
                    for pt in pts:
                        pl = (getattr(pt, "payload", {}) or {})
                        md = pl.get("metadata") or {}
                        path = md.get("path")
                        content = pl.get("content") or pl.get("text") or pl.get("information") or md.get("information")
                        if not path and content:
                            hits += 1
                    if hits > best_hits:
                        best_hits = hits
                        best_name = name
                except Exception:
                    continue
            if best_name and best_hits > 0:
                mcoll = best_name
        except Exception:
            pass

    try:
        lim = int(limit) if (limit is not None and str(limit).strip() != "") else 10
    except Exception:
        lim = 10
    try:
        per_path_val = int(per_path) if (per_path is not None and str(per_path).strip() != "") else 1
    except Exception:
        per_path_val = 1

    # Normalize queries to list (accept q/text aliases)
    queries: List[str] = []
    if (query is None or (isinstance(query, str) and query.strip() == "")):
        q_alt = kwargs.get("q") or kwargs.get("text")
        if q_alt is not None:
            query = q_alt
    if isinstance(query, (list, tuple)):
        queries = [str(q) for q in query]
    elif query is not None and str(query).strip() != "":
        queries = [str(query)]

    # Accept common alias keys and camelCase from clients
    if (limit is None or (isinstance(limit, str) and limit.strip() == "")) and ("top_k" in kwargs):
        limit = kwargs.get("top_k")
    if include_memories is None and ("includeMemories" in kwargs):
        include_memories = kwargs.get("includeMemories")
    if memory_weight is None and ("memoryWeight" in kwargs):
        memory_weight = kwargs.get("memoryWeight")
    if per_source_limits is None and ("perSourceLimits" in kwargs):
        per_source_limits = kwargs.get("perSourceLimits")

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


    # Option A: Query the memory MCP server over SSE and blend results (real integration)
    mem_hits: List[Dict[str, Any]] = []
    memory_note: str = ""
    if include_mem and mem_limit > 0 and queries and use_sse_memory:
        try:
            # Import the FastMCP client if available; record a helpful note otherwise
            try:
                from fastmcp import Client  # use FastMCP client for SSE interop
            except ImportError:
                memory_note = "SSE memory disabled: fastmcp client not installed"
                raise
            import asyncio
            timeout = float(os.environ.get("MEMORY_MCP_TIMEOUT", "6"))
            base_url = os.environ.get("MEMORY_MCP_URL") or "http://mcp:8000/sse"
            async with Client(base_url) as c:
                tools = await asyncio.wait_for(c.list_tools(), timeout=timeout)
                tool_name = None
                # Prefer canonical names
                for t in tools:
                    tn = (getattr(t, "name", None) or "").strip()
                    tl = tn.lower()
                    if tl in ("find", "memory.find"):
                        tool_name = tn
                        break
                if tool_name is None:
                    for t in tools:
                        tn = (getattr(t, "name", None) or "").strip()
                        if "find" in tn.lower():
                            tool_name = tn
                            break
                if tool_name:
                    qtext = " ".join([q for q in queries if q]).strip() or queries[0]
                    arg_variants: List[Dict[str, Any]] = [
                        {"query": qtext, "limit": mem_limit},
                        {"q": qtext, "limit": mem_limit},
                        {"text": qtext, "limit": mem_limit},
                    ]
                    res_obj = None
                    for args in arg_variants:
                        try:
                            res_obj = await asyncio.wait_for(c.call_tool(tool_name, args), timeout=timeout)
                            break
                        except Exception:
                            continue
                    if res_obj is not None:
                        # Normalize FastMCP result content -> rd-like dict
                        rd = {"content": []}
                        try:
                            for item in getattr(res_obj, "content", []) or []:
                                txt = getattr(item, "text", None)
                                if isinstance(txt, str):
                                    rd["content"].append({"type": "text", "text": txt})
                        except Exception:
                            rd = {}
                        # Parse common MCP tool result shapes
                        def push_text(txt: str, md: Dict[str, Any] | None = None, score: float | int | None = None):
                            if not txt:
                                return
                            mem_hits.append({
                                "source": "memory",
                                "score": float(score or 1.0),
                                "content": txt,
                                "metadata": (md or {}),
                            })
                        if isinstance(rd, dict):
                            cont = rd.get("content")
                            if isinstance(cont, list):
                                for c in cont:
                                    try:
                                        ctype = c.get("type")
                                        if ctype == "text" and isinstance(c.get("text"), str):
                                            push_text(c["text"], {})
                                        elif ctype == "json":
                                            j = c.get("json")
                                            if isinstance(j, list):
                                                for it in j:
                                                    if isinstance(it, dict):
                                                        push_text(
                                                            str(it.get("text") or it.get("content") or it.get("information") or ""),
                                                            it.get("metadata") or {},
                                                            it.get("score") or 1.0,
                                                        )
                                            elif isinstance(j, dict):
                                                items = j.get("results") or j.get("items") or j.get("memories") or j.get("data")
                                                if isinstance(items, list):
                                                    for it in items:
                                                        if isinstance(it, dict):
                                                            push_text(
                                                                str(it.get("text") or it.get("content") or it.get("information") or ""),
                                                                it.get("metadata") or {},
                                                                it.get("score") or 1.0,
                                                            )
                                    except Exception:
                                        continue
                            # Fallback if provider returns flat dict
                            if not mem_hits:
                                items = rd.get("results") or rd.get("items")
                                if isinstance(items, list):
                                    for it in items:
                                        if isinstance(it, dict):
                                            push_text(
                                                str(it.get("text") or it.get("content") or it.get("information") or ""),
                                                it.get("metadata") or {},
                                                it.get("score") or 1.0,
                                            )
        except Exception:
            pass

    # If SSE memory didn’t yield hits, try local Qdrant memory-like retrieval as fallback
    if include_mem and mem_limit > 0 and not mem_hits and queries:
        try:
            from qdrant_client import QdrantClient  # type: ignore

            from scripts.utils import sanitize_vector_name  # local util

            client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            vec_name = sanitize_vector_name(model_name)
            model = _get_embedding_model(model_name)

            qtext = " ".join([q for q in queries if q]).strip() or queries[0]
            v = next(model.embed([qtext])).tolist()
            k = max(mem_limit, 5)
            res = await asyncio.to_thread(lambda: client.search(
                collection_name=mcoll,
                query_vector={"name": vec_name, "vector": v},
                limit=k,
                with_payload=True,
            ))
            for pt in res:
                payload = (getattr(pt, "payload", {}) or {})
                md = payload.get("metadata") or {}
                path = str(md.get("path") or "")
                start_line = md.get("start_line")
                end_line = md.get("end_line")
                content = payload.get("content") or payload.get("text") or payload.get("information") or md.get("information")
                kind = (md.get("kind") or payload.get("kind") or "").lower()
                source_tag = (md.get("source") or payload.get("source") or "").lower()
                flagged = kind in ("memory","preference","note","policy","infra","chat") or source_tag in ("memory","chat")
                is_memory_like = (not path) or (start_line in (None, 0) and end_line in (None, 0)) or flagged
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
            import re
            terms = [str(t).lower() for t in queries if t]
            tokens = set()
            for t in terms:
                tokens.update([w for w in re.split(r"[^a-z0-9_]+", t) if len(w) >= 3])
            if not tokens:
                tokens = set(terms)
            checked = 0
            cap = 2000
            page = None
            while len(mem_hits) < mem_limit and checked < cap:
                sc, page = await asyncio.to_thread(
                    lambda: client.scroll(
                        collection_name=mcoll,
                        with_payload=True,
                        with_vectors=False,
                        limit=500,
                        offset=page,
                    )
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
                    kind = (md.get("kind") or payload.get("kind") or "").lower()
                    source_tag = (md.get("source") or payload.get("source") or "").lower()
                    flagged = kind in ("memory","preference","note","policy","infra","chat") or source_tag in ("memory","chat")
                    is_memory_like = (not path) or (start_line in (None, 0) and end_line in (None, 0)) or flagged
                    if not (is_memory_like and content):
                        continue
                    low = str(content).lower()
                    if any(tok in low for tok in tokens):
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
        ret = {"results": compacted, "total": len(compacted)}
        if memory_note:
            ret["memory_note"] = memory_note
        return ret

    ret = {"results": blended, "total": len(blended)}
    if memory_note:
        ret["memory_note"] = memory_note
    return ret

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
    **kwargs,
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
        **kwargs,
    )



if __name__ == "__main__":

    # Optional warmups: gated by env flags to avoid delaying readiness on fresh containers
    try:
        if str(os.environ.get("EMBEDDING_WARMUP", "")).strip().lower() in {"1","true","yes","on"}:
            _ = _get_embedding_model(os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))
    except Exception:
        pass
    try:
        if str(os.environ.get("RERANK_WARMUP", "")).strip().lower() in {"1","true","yes","on"} and \
           str(os.environ.get("RERANKER_ENABLED", "")).strip().lower() in {"1","true","yes","on"}:
            if str(os.environ.get("RERANK_IN_PROCESS", "")).strip().lower() in {"1","true","yes","on"}:
                try:
                    from scripts.rerank_local import _get_rerank_session  # type: ignore
                    _ = _get_rerank_session()
                except Exception:
                    pass
            else:
                # Fire a tiny warmup rerank once via subprocess; ignore failures
                _env = os.environ.copy()
                _env["QDRANT_URL"] = QDRANT_URL
                _env["COLLECTION_NAME"] = DEFAULT_COLLECTION
                _cmd = ["python", "/work/scripts/rerank_local.py", "--query", "warmup", "--topk", "3", "--limit", "1"]
                subprocess.run(_cmd, capture_output=True, text=True, env=_env, timeout=10)
    except Exception:
        pass

    transport = os.environ.get("FASTMCP_TRANSPORT", "sse").strip().lower()
    if transport == "stdio":
        # Run over stdio (for clients that don't support SSE)
        mcp.run(transport="stdio")
    else:
        # Serve over SSE at /sse on the configured host/port
        mcp.settings.host = HOST
        mcp.settings.port = PORT
        mcp.run(transport="sse")

