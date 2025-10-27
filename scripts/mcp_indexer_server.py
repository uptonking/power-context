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

import sys

# Ensure code roots are on sys.path so absolute imports like 'from scripts.x import y' work
# when this file is executed directly (sys.path[0] may be /work/scripts).
# Supports multiple roots via WORK_ROOTS env (comma-separated), defaults to /work and /app.
_roots_env = os.environ.get("WORK_ROOTS", "")

# Cache for memory collection autodetection (name + timestamp)
_MEM_COLL_CACHE = {"name": None, "ts": 0.0}

_roots = [p.strip() for p in _roots_env.split(",") if p.strip()] or ["/work", "/app"]
try:
    for _root in _roots:
        if _root and _root not in sys.path:
            sys.path.insert(0, _root)
except Exception:
    pass


# Shared utilities (lex hashing, snippet highlighter)
try:
    from scripts.utils import highlight_snippet as _do_highlight_snippet
except Exception:
    _do_highlight_snippet = None  # fallback guarded at call site


# Back-compat shim for tests expecting _highlight_snippet in this module
# Delegates to scripts.utils.highlight_snippet when available
try:
    def _highlight_snippet(snippet, tokens):  # type: ignore
        return _do_highlight_snippet(snippet, tokens) if _do_highlight_snippet else snippet
except Exception:
    def _highlight_snippet(snippet, tokens):  # type: ignore
        return snippet

try:
    # Official MCP Python SDK (FastMCP convenience server)
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    raise SystemExit("mcp package is required inside the container: pip install mcp")

APP_NAME = os.environ.get("FASTMCP_SERVER_NAME", "qdrant-indexer-mcp")
HOST = os.environ.get("FASTMCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("FASTMCP_INDEXER_PORT", "8001"))

# Process-wide lock to guard environment mutations during retrieval (gate-first/budgeting)
_ENV_LOCK = threading.RLock()


def _primary_identifier_from_queries(qs: list[str]) -> str:
    """Best-effort extraction of the main CONSTANT_NAME or IDENTIFIER from queries."""
    try:
        import re as _re
        cand: list[str] = []
        for q in qs:
            for t in _re.findall(r"[A-Za-z_][A-Za-z0-9_]*", q or ""):
                if len(t) >= 2 and ("_" in t or t.isupper()):
                    cand.append(t)
        return next((t for t in cand if t), "")
    except Exception:
        return ""

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
MAX_LOG_TAIL = int(os.environ.get("MCP_MAX_LOG_TAIL", "4000"))
SNIPPET_MAX_BYTES = int(os.environ.get("MCP_SNIPPET_MAX_BYTES", "8192") or 8192)

MCP_TOOL_TIMEOUT_SECS = float(os.environ.get("MCP_TOOL_TIMEOUT_SECS", "3600") or 3600.0)

# --- Workspace state integration helpers ---
def _state_file_path(ws_path: str = "/work") -> str:
    try:
        return os.path.join(ws_path, ".codebase", "state.json")
    except Exception:
        return "/work/.codebase/state.json"


def _read_ws_state(ws_path: str = "/work") -> Optional[Dict[str, Any]]:
    try:
        p = _state_file_path(ws_path)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _default_collection() -> str:
    st = _read_ws_state("/work")
    if st:
        coll = st.get("qdrant_collection")
        if isinstance(coll, str) and coll.strip():
            return coll.strip()
    return DEFAULT_COLLECTION



def _work_script(name: str) -> str:
    """Return path to a script under /work if present, else local ./scripts.
    Keeps Docker/default behavior but works in local dev without /work mount.
    """
    try:
        w = os.path.join("/work", "scripts", name)
        if os.path.exists(w):
            return w
    except Exception:
        pass
    return os.path.join(os.getcwd(), "scripts", name)


# Invalidate router scratchpad after reindex to avoid stale state reuse
_def_ws = "/work"

def _invalidate_router_scratchpad(ws_path: str = _def_ws) -> bool:
    try:
        p = os.path.join(ws_path, ".codebase", "router_scratchpad.json")
        if os.path.exists(p):
            os.remove(p)
            return True
    except Exception:
        pass
    return False


mcp = FastMCP(APP_NAME)


# Capture tool registry automatically by wrapping the decorator once
_TOOLS_REGISTRY: list[dict] = []
try:
    _orig_tool = mcp.tool
    def _tool_capture_wrapper(*dargs, **dkwargs):
        orig_deco = _orig_tool(*dargs, **dkwargs)
        def _inner(fn):
            try:
                _TOOLS_REGISTRY.append({
                    "name": dkwargs.get("name") or getattr(fn, "__name__", ""),
                    "description": (getattr(fn, "__doc__", None) or "").strip(),
                })
            except Exception:
                pass
            return orig_deco(fn)
        return _inner
    mcp.tool = _tool_capture_wrapper  # type: ignore
except Exception:
    pass

# Lightweight readiness endpoint on a separate health port (non-MCP), optional
# Exposes GET /readyz returning {ok: true, app: <name>} once process is up.
try:
    HEALTH_PORT = int(os.environ.get("FASTMCP_HEALTH_PORT", "18001") or 18001)
except Exception:
    HEALTH_PORT = 18001


def _start_readyz_server():
    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == "/readyz":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        payload = {"ok": True, "app": APP_NAME}
                        self.wfile.write((json.dumps(payload)).encode("utf-8"))
                    elif self.path == "/tools":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        # Hide expand_query when decoder is disabled
                        tools = _TOOLS_REGISTRY
                        try:
                            from scripts.refrag_llamacpp import is_decoder_enabled  # type: ignore
                        except Exception:
                            is_decoder_enabled = lambda: False  # type: ignore
                        try:
                            if not is_decoder_enabled():
                                tools = [t for t in tools if (t.get("name") or "") != "expand_query"]
                        except Exception:
                            pass
                        payload = {"ok": True, "tools": tools}
                        self.wfile.write((json.dumps(payload)).encode("utf-8"))
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception:
                    try:
                        self.send_response(500)
                        self.end_headers()
                    except Exception:
                        pass



            def log_message(self, *args, **kwargs):
                # Quiet health server logs
                return

        srv = HTTPServer((HOST, HEALTH_PORT), H)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        return True
    except Exception:
        return False


# Async subprocess runner to avoid blocking event loop
async def _run_async(
    cmd: list[str], env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None
) -> Dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # Default timeout from env if not provided by caller
        if timeout is None:
            timeout = MCP_TOOL_TIMEOUT_SECS
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
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
            return (
                s
                if len(s) <= MAX_LOG_TAIL
                else ("...[tail truncated]\n" + s[-MAX_LOG_TAIL:])
            )

        return {
            "ok": code == 0,
            "code": code,
            "stdout": _cap_tail(stdout),
            "stderr": _cap_tail(stderr),
        }
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
        pass


# Extra parsing helpers for quirky clients that send stringified kwargs
import urllib.parse as _urlparse, ast as _ast


def _parse_kv_string(s: str) -> _Dict[str, _Any]:
    """Parse non-JSON strings like "a=1&b=2" or "query=[\"a\",\"b\"]" into a dict.
    Values are JSON-decoded when possible; else literal-eval; else kept as raw strings.
    """
    out: _Dict[str, _Any] = {}
    try:
        if not isinstance(s, str) or not s.strip():
            return out
        # Try query-string form first
        if ("=" in s) and ("{" not in s) and (":" not in s):
            qs = _urlparse.parse_qs(s, keep_blank_values=True)
            for k, vals in qs.items():
                v = vals[-1] if vals else ""
                out[k] = _coerce_value_string(v)
            return out
        # Fallback: split on commas for simple "k=v,k2=v2" forms
        if ("=" in s) and ("," in s):
            for part in s.split(","):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                out[k.strip()] = _coerce_value_string(v.strip())
            return out
    except Exception:
        return {}
    return out


def _coerce_value_string(v: str):
    # Try JSON
    try:
        return json.loads(v)
    except Exception:
        pass
    # Try Python literal (e.g., "['a','b']")
    try:
        return _ast.literal_eval(v)
    except Exception:
        pass
    # As-is string
    return v


def _to_str_list_relaxed(x: _Any) -> list[str]:
    """Coerce various inputs to list[str]. Accepts JSON strings like "[\"a\",\"b\"]"."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(e) for e in x if str(e).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Try JSON array or Python literal list
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(e) for e in arr if str(e).strip()]
            except Exception:
                try:
                    arr = _ast.literal_eval(s)
                    if isinstance(arr, (list, tuple)):
                        return [str(e) for e in arr if str(e).strip()]
                except Exception:
                    pass
        # Comma-separated fallback
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return [str(x)]


def _extract_kwargs_payload(kwargs: _Any) -> _Dict[str, _Any]:
    try:
        if isinstance(kwargs, dict) and "kwargs" in kwargs:
            inner = kwargs.get("kwargs")
            if isinstance(inner, dict):
                return inner
            parsed = _maybe_parse_jsonish(inner)
            if isinstance(parsed, dict):
                return parsed
            # Fallback: accept query-string or k=v,k2=v2 strings
            if isinstance(inner, str):
                kv = _parse_kv_string(inner)
                if isinstance(kv, dict) and kv:
                    return kv
            return {}
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

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "to",
    "with",
    "by",
    "is",
    "are",
    "be",
    "this",
    "that",
}


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
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


@mcp.tool()
async def qdrant_index_root(
    recreate: Optional[bool] = None, collection: Optional[str] = None
) -> Dict[str, Any]:
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

    # Resolve collection: prefer explicit non-placeholder; otherwise workspace state
    try:
        _c = (collection or "").strip()
    except Exception:
        _c = ""
    _placeholders = {"", "my-collection"}
    if _c and _c not in _placeholders:
        coll = _c
    else:
        try:
            from scripts.workspace_state import get_collection_name as _ws_get_collection_name  # type: ignore
            coll = _ws_get_collection_name("/work")
        except Exception:
            coll = _default_collection()

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = ["python", _work_script("ingest_code.py"), "--root", "/work"]
    if recreate:
        cmd.append("--recreate")

    res = await _run_async(cmd, env=env)
    ret = {"args": {"root": "/work", "collection": coll, "recreate": recreate}, **res}
    try:
        if ret.get("ok") and int(ret.get("code", 1)) == 0:
            if _invalidate_router_scratchpad("/work"):
                ret["invalidated_router_scratchpad"] = True
    except Exception:
        pass
    return ret


@mcp.tool()
async def qdrant_list(**kwargs) -> Dict[str, Any]:
    """List Qdrant collections (ignores any extra params)"""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        cols_info = await asyncio.to_thread(client.get_collections)
        return {"collections": [c.name for c in cols_info.collections]}
    except ImportError:
        return {"error": "qdrant_client is not installed in this container"}
    except Exception as e:
        return {"error": str(e)}



@mcp.tool()
async def workspace_info(workspace_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Return the current workspace state from .codebase/state.json, if present.
    Returns:
      - workspace_path: resolved path (defaults to "/work")
      - default_collection: resolved collection (state.json > env > DEFAULT)
      - source: "state_file" or "env"
      - state: raw state.json contents (or {})
    """
    ws_path = (workspace_path or "/work").strip() or "/work"
    st = _read_ws_state(ws_path) or {}
    coll = (st.get("qdrant_collection") if isinstance(st, dict) else None) or os.environ.get("COLLECTION_NAME") or DEFAULT_COLLECTION
    return {
        "workspace_path": ws_path,
        "default_collection": coll,
        "source": ("state_file" if st else "env"),
        "state": st or {},
    }

@mcp.tool()
async def list_workspaces(search_root: Optional[str] = None) -> Dict[str, Any]:
    """Discover workspaces by scanning for .codebase/state.json files.
    Returns: {"workspaces": [{"workspace_path", "collection_name", "last_updated", "indexing_state"}, ...]}
    """
    try:
        from scripts.workspace_state import list_workspaces as _lw  # type: ignore
        items = await asyncio.to_thread(lambda: _lw(search_root))
        return {"workspaces": items}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def memory_store(
    information: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
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
        from scripts.ingest_code import project_mini as _project_mini  # type: ignore

    except Exception as e:  # pragma: no cover
        return {"error": f"deps: {e}"}

    if not information or not str(information).strip():
        return {"error": "information is required"}

    coll = (collection or _default_collection()) or ""
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
        # Delegate to shared utility for consistency
        try:
            from scripts.utils import lex_hash_vector_text

            return lex_hash_vector_text(text, dim)
        except Exception:
            # Fallback: minimal hashing
            if not text:
                return [0.0] * dim
            vec = [0.0] * dim
            toks = _split_ident_lex(text)
            if not toks:
                return vec
            for t in toks:
                h = int(
                    hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16
                )
                vec[h % dim] += 1.0
            norm = (sum(v * v for v in vec) or 0.0) ** 0.5 or 1.0
            return [v / norm for v in vec]

    # Build vectors (cached embedding model)
    model = _get_embedding_model(model_name)
    dense = next(model.embed([str(information)])).tolist()

    lex = _lex_hash_vector(str(information))

    # Upsert
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        # Ensure collection and named vectors exist (dense + lexical)
        try:
            await asyncio.to_thread(
                lambda: _ensure_collection(client, coll, len(dense), vector_name)
            )
        except Exception:
            pass
        pid = str(uuid.uuid4())
        payload = {
            "information": str(information),
            "metadata": metadata or {"kind": "memory", "source": "memory"},
        }
        vecs = {vector_name: dense, LEX_VECTOR_NAME: lex}
        try:
            if str(os.environ.get("REFRAG_MODE", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                mini_name = os.environ.get("MINI_VECTOR_NAME", "mini")
                mini = _project_mini(
                    list(dense), int(os.environ.get("MINI_VEC_DIM", "64") or 64)
                )
                vecs[mini_name] = mini
        except Exception:
            pass
        point = models.PointStruct(id=pid, vector=vecs, payload=payload)
        await asyncio.to_thread(
            lambda: client.upsert(collection_name=coll, points=[point], wait=True)
        )
        return {"ok": True, "id": pid, "collection": coll, "vector_name": vector_name}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def qdrant_status(
    collection: Optional[str] = None,
    max_points: Optional[int] = None,
    batch: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
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
    coll = collection or _default_collection()
    try:
        from qdrant_client import QdrantClient
        import datetime as _dt

        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        # Count points
        try:
            cnt_res = await asyncio.to_thread(
                lambda: client.count(collection_name=coll, exact=True)
            )
            total = int(getattr(cnt_res, "count", 0))
        except Exception:
            total = 0
        # Scan a limited number of points to estimate last timestamps
        max_points = (
            int(max_points)
            if max_points not in (None, "")
            else int(os.environ.get("MCP_STATUS_MAX_POINTS", "5000"))
        )
        batch = int(batch) if batch not in (None, "") else 1000
        scanned = 0
        last_ing = None
        last_mod = None
        next_page = None
        while scanned < max_points:
            limit = min(batch, max_points - scanned)
            try:
                pts, next_page = await asyncio.to_thread(
                    lambda: client.scroll(
                        collection_name=coll,
                        limit=limit,
                        offset=next_page,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
            except Exception:
                # Fallback without offset keyword (older clients)
                pts, next_page = await asyncio.to_thread(
                    lambda: client.scroll(
                        collection_name=coll,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
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
async def qdrant_index(
    subdir: Optional[str] = None,
    recreate: Optional[bool] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
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
    # Resolve collection: prefer explicit non-placeholder; otherwise workspace state (use workspace root)
    try:
        _c2 = (collection or "").strip()
    except Exception:
        _c2 = ""
    _placeholders2 = {"", "my-collection"}
    if _c2 and _c2 not in _placeholders2:
        coll = _c2
    else:
        try:
            from scripts.workspace_state import get_collection_name as _ws_get_collection_name  # type: ignore
            coll = _ws_get_collection_name("/work")
        except Exception:
            coll = _default_collection()

    env = os.environ.copy()
    env["QDRANT_URL"] = QDRANT_URL
    env["COLLECTION_NAME"] = coll

    cmd = [
        "python",
        _work_script("ingest_code.py"),
        "--root",
        root,
    ]
    if recreate:
        cmd.append("--recreate")

    res = await _run_async(cmd, env=env)
    ret = {"args": {"root": root, "collection": coll, "recreate": recreate}, **res}
    try:
        if ret.get("ok") and int(ret.get("code", 1)) == 0:
            if _invalidate_router_scratchpad("/work"):
                ret["invalidated_router_scratchpad"] = True
    except Exception:
        pass
    return ret


@mcp.tool()
async def qdrant_prune(**kwargs) -> Dict[str, Any]:
    """Prune stale points for the mounted path (/work). Extra params are ignored."""
    env = os.environ.copy()
    env["PRUNE_ROOT"] = "/work"

    cmd = ["python", _work_script("prune.py")]
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
    workspace_path: Any = None,
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
        if (
            limit is None or (isinstance(limit, str) and str(limit).strip() == "")
        ) and ("top_k" in kwargs):
            limit = kwargs.get("top_k")
        if query is None or (isinstance(query, str) and str(query).strip() == ""):
            q_alt = kwargs.get("q") or kwargs.get("text")
            if q_alt is not None:
                query = q_alt
    except Exception:
        pass

    # Leniency: absorb nested 'kwargs' JSON payload some clients send
    try:
        _extra = _extract_kwargs_payload(kwargs)
        if _extra:
            if query is None or (isinstance(query, str) and query.strip() == ""):
                query = _extra.get("query") or _extra.get("queries")
            if limit in (None, "") and _extra.get("limit") is not None:
                limit = _extra.get("limit")
            if per_path in (None, "") and _extra.get("per_path") is not None:
                per_path = _extra.get("per_path")
            if (
                include_snippet in (None, "")
                and _extra.get("include_snippet") is not None
            ):
                include_snippet = _extra.get("include_snippet")
            if context_lines in (None, "") and _extra.get("context_lines") is not None:
                context_lines = _extra.get("context_lines")
            if (
                rerank_enabled in (None, "")
                and _extra.get("rerank_enabled") is not None
            ):
                rerank_enabled = _extra.get("rerank_enabled")
            if rerank_top_n in (None, "") and _extra.get("rerank_top_n") is not None:
                rerank_top_n = _extra.get("rerank_top_n")
            if (
                rerank_return_m in (None, "")
                and _extra.get("rerank_return_m") is not None
            ):
                rerank_return_m = _extra.get("rerank_return_m")
            if (
                rerank_timeout_ms in (None, "")
                and _extra.get("rerank_timeout_ms") is not None
            ):
                rerank_timeout_ms = _extra.get("rerank_timeout_ms")
            if (
                highlight_snippet in (None, "")
                and _extra.get("highlight_snippet") is not None
            ):
                highlight_snippet = _extra.get("highlight_snippet")
            if (
                collection is None
                or (isinstance(collection, str) and collection.strip() == "")
            ) and _extra.get("collection"):
                collection = _extra.get("collection")
            # Optional workspace_path routing
            if (
                (workspace_path is None) or (isinstance(workspace_path, str) and str(workspace_path).strip() == "")
            ) and _extra.get("workspace_path") is not None:
                workspace_path = _extra.get("workspace_path")
            if (
                language is None
                or (isinstance(language, str) and language.strip() == "")
            ) and _extra.get("language"):
                language = _extra.get("language")
            if (
                under is None or (isinstance(under, str) and under.strip() == "")
            ) and _extra.get("under"):
                under = _extra.get("under")
            if (
                kind is None or (isinstance(kind, str) and kind.strip() == "")
            ) and _extra.get("kind"):
                kind = _extra.get("kind")
            if (
                symbol is None or (isinstance(symbol, str) and symbol.strip() == "")
            ) and _extra.get("symbol"):
                symbol = _extra.get("symbol")
            if (
                path_regex is None
                or (isinstance(path_regex, str) and path_regex.strip() == "")
            ) and _extra.get("path_regex"):
                path_regex = _extra.get("path_regex")
            if path_glob in (None, "") and _extra.get("path_glob") is not None:
                path_glob = _extra.get("path_glob")
            if not_glob in (None, "") and _extra.get("not_glob") is not None:
                not_glob = _extra.get("not_glob")
            if (
                ext is None or (isinstance(ext, str) and ext.strip() == "")
            ) and _extra.get("ext"):
                ext = _extra.get("ext")
            if (not_ is None or (isinstance(not_, str) and not_.strip() == "")) and (
                _extra.get("not") or _extra.get("not_")
            ):
                not_ = _extra.get("not") or _extra.get("not_")
            if (
                case is None or (isinstance(case, str) and case.strip() == "")
            ) and _extra.get("case"):
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
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
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
    # Reranker: default ON; can be disabled via env or client args
    rerank_env_default = str(
        os.environ.get("RERANKER_ENABLED", "1")
    ).strip().lower() in {"1", "true", "yes", "on"}
    rerank_enabled = _to_bool(rerank_enabled, rerank_env_default)
    rerank_top_n = _to_int(
        rerank_top_n, int(os.environ.get("RERANKER_TOPN", "50") or 50)
    )
    rerank_return_m = _to_int(
        rerank_return_m, int(os.environ.get("RERANKER_RETURN_M", "12") or 12)
    )
    rerank_timeout_ms = _to_int(
        rerank_timeout_ms, int(os.environ.get("RERANKER_TIMEOUT_MS", "120") or 120)
    )
    highlight_snippet = _to_bool(highlight_snippet, True)

    # Resolve collection: explicit > workspace_path state > default
    ws_hint = _to_str(workspace_path, "").strip()
    coll_hint = _to_str(collection, "").strip()
    if not coll_hint and ws_hint:
        try:
            st = _read_ws_state(ws_hint)
            if st and isinstance(st.get("qdrant_collection"), str):
                coll_hint = st.get("qdrant_collection").strip()
        except Exception:
            pass
    collection = coll_hint or _default_collection()

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
    # If snippets are requested, do not compact (we need snippet field in results)
    if include_snippet:
        compact = False

    # Accept top-level alias `queries` as a drop-in for `query`
    # Many clients send queries=[...] instead of query=[...]
    if "queries" in kwargs and kwargs.get("queries") is not None:
        query = kwargs.get("queries")

    # Normalize queries to a list[str] (robust for JSON strings and arrays)
    queries: list[str] = []
    if isinstance(query, (list, tuple)):
        queries = [str(q).strip() for q in query if str(q).strip()]
    elif isinstance(query, str):
        queries = _to_str_list_relaxed(query)
    elif query is not None:
        s = str(query).strip()
        if s:
            queries = [s]

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

    use_hybrid_inproc = str(
        os.environ.get("HYBRID_IN_PROCESS", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if use_hybrid_inproc:
        try:
            from scripts.hybrid_search import run_hybrid_search  # type: ignore

            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            model = _get_embedding_model(model_name)
            # In-process path_glob/not_glob accept a single string; reduce list inputs safely
            items = run_hybrid_search(
                queries=queries,
                limit=int(limit),
                per_path=(int(per_path) if (per_path is not None and str(per_path).strip() != "") else 1),
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
                expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower()
                in {"1", "true", "yes", "on"},
                model=model,
            )
            # items are already in structured dict form
            json_lines = items  # reuse downstream shaping
        except Exception as e:
            # Fallback to subprocess path if in-process fails
            use_hybrid_inproc = False

    if not use_hybrid_inproc:
        # Try hybrid search via subprocess (JSONL output)
        cmd = [
            "python",
            _work_script("hybrid_search.py"),
            "--limit",
            str(int(limit)),
            "--json",
        ]
        if (per_path is not None and str(per_path).strip() != ""):
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
        # Fallback: if subprocess yielded nothing (e.g., local dev without /work), try in-process once
        if (not json_lines):
            try:
                from scripts.hybrid_search import run_hybrid_search  # type: ignore

                model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                model = _get_embedding_model(model_name)
                items = run_hybrid_search(
                    queries=queries,
                    limit=int(limit),
                    per_path=(int(per_path) if (per_path is not None and str(per_path).strip() != "") else 1),
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
                    expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower() in {"1", "true", "yes", "on"},
                    model=model,
                )
                json_lines = items
            except Exception:
                pass

    # Optional rerank fallback path: if enabled, attempt; on timeout or error, keep hybrid
    used_rerank = False
    rerank_counters = {
        "inproc_hybrid": 0,
        "inproc_dense": 0,
        "subprocess": 0,
        "timeout": 0,
        "error": 0,
    }
    if rerank_enabled:
        # Resolve in-process gating once and reuse
        use_rerank_inproc = str(
            os.environ.get("RERANK_IN_PROCESS", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Prefer fusion-aware reranking over hybrid candidates when available, but only if in-process reranker is enabled
        if use_rerank_inproc:
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
                            with open(
                                realp, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                lines = f.readlines()
                            ctx = (
                                max(1, int(context_lines))
                                if "context_lines" in locals()
                                else 2
                            )
                            si = max(1, sl - ctx)
                            ei = min(len(lines), max(sl, el) + ctx)
                            snippet = "".join(lines[si - 1 : ei]).strip()
                            return (
                                header + ("\n" + snippet if snippet else "")
                            ).strip()
                        except Exception:
                            return header

                    # Build docs concurrently
                    max_workers = min(8, (os.cpu_count() or 4) * 2)
                    with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
                        docs = list(ex.map(_doc_for, cand_objs))
                    pairs = [(rq, d) for d in docs]
                    scores = _rr_local(pairs)
                    ranked = sorted(
                        zip(scores, cand_objs), key=lambda x: x[0], reverse=True
                    )
                    tmp = []
                    for s, obj in ranked[: int(rerank_return_m)]:
                        item = {
                            "score": float(s),
                            "path": obj.get("path", ""),
                            "symbol": obj.get("symbol", ""),
                            "start_line": int(obj.get("start_line") or 0),
                            "end_line": int(obj.get("end_line") or 0),
                            "why": obj.get("why", []) + [f"rerank_onnx:{float(s):.3f}"],
                            "components": (obj.get("components") or {})
                            | {"rerank_onnx": float(s)},
                        }
                        tmp.append(item)
                    if tmp:
                        results = tmp
                        used_rerank = True
                        rerank_counters["inproc_hybrid"] += 1
            except Exception:
                used_rerank = False
        # Fallback paths (in-process reranker dense candidates, then subprocess)
        if not used_rerank:
            if use_rerank_inproc:
                try:
                    from scripts.rerank_local import rerank_in_process  # type: ignore

                    model_name = os.environ.get(
                        "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
                    )
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
                        rerank_counters["inproc_dense"] += 1
                except Exception:
                    use_rerank_inproc = False
            if (not use_rerank_inproc) and (not used_rerank):
                try:
                    rq = queries[0] if queries else ""
                    rcmd = [
                        "python",
                        _work_script("rerank_local.py"),
                        "--query",
                        rq,
                        "--topk",
                        str(int(rerank_top_n)),
                        "--limit",
                        str(int(rerank_return_m)),
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
                            print(
                                "RERANK_RET:",
                                rres.get("code"),
                                "OUT_LEN:",
                                len((rres.get("stdout") or "").strip()),
                                "ERR_TAIL:",
                                (rres.get("stderr") or "")[-200:],
                            )
                        except Exception:
                            pass
                    if rres.get("ok") and (rres.get("stdout") or "").strip():
                        rerank_counters["subprocess"] += 1
                        tmp = []
                        for ln in (rres.get("stdout") or "").splitlines():
                            parts = ln.strip().split("\t")
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
                            item = {
                                "score": score,
                                "path": path,
                                "symbol": symbol,
                                "start_line": start_line,
                                "end_line": end_line,
                                "why": [f"rerank_onnx:{score:.3f}"],
                            }
                            tmp.append(item)
                        if tmp:
                            results = tmp
                            used_rerank = True
                            rerank_counters["inproc_hybrid"] += 1
                except subprocess.TimeoutExpired:
                    rerank_counters["timeout"] += 1
                    used_rerank = False
                except Exception:
                    rerank_counters["error"] += 1
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
            # Pass-through optional relation hints
            if obj.get("relations"):
                item["relations"] = obj.get("relations")
            if obj.get("related_paths"):
                item["related_paths"] = obj.get("related_paths")
            if obj.get("span_budgeted") is not None:
                item["span_budgeted"] = bool(obj.get("span_budgeted"))
            if obj.get("budget_tokens_used") is not None:
                item["budget_tokens_used"] = int(obj.get("budget_tokens_used"))
            results.append(item)

    # Optionally add snippets (with highlighting)
    toks = _tokens_from_queries(queries)
    if include_snippet:
        import concurrent.futures as _fut

        def _read_snip(args):
            i, item = args
            try:
                path = item.get("path")
                sl = int(item.get("start_line") or 0)
                el = int(item.get("end_line") or 0)
                if not path or not sl:
                    return (i, "")
                raw_path = str(path)
                p = (
                    raw_path
                    if os.path.isabs(raw_path)
                    else os.path.join("/work", raw_path)
                )
                realp = os.path.realpath(p)
                if not (realp == "/work" or realp.startswith("/work/")):
                    return (i, "")
                with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                ctx = max(1, int(context_lines))
                si = max(1, sl - ctx)
                ei = min(len(lines), max(sl, el) + ctx)
                snippet = "".join(lines[si - 1 : ei])
                if highlight_snippet:
                    snippet = (
                        _do_highlight_snippet(snippet, toks)
                        if _do_highlight_snippet
                        else snippet
                    )
                if len(snippet.encode("utf-8", "ignore")) > SNIPPET_MAX_BYTES:
                    _suffix = "\n...[snippet truncated]"
                    _sb = _suffix.encode("utf-8")
                    _bytes = snippet.encode("utf-8", "ignore")
                    _keep = max(0, SNIPPET_MAX_BYTES - len(_sb))
                    _trimmed = _bytes[:_keep]
                    snippet = _trimmed.decode("utf-8", "ignore") + _suffix
                return (i, snippet)
            except Exception:
                return (i, "")

        max_workers = min(8, (os.cpu_count() or 4) * 2)
        with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, snip in ex.map(_read_snip, list(enumerate(results))):
                try:
                    results[i]["snippet"] = snip
                except Exception:
                    pass

    # Smart default: compact true for multi-query calls if compact not explicitly set
    if (len(queries) > 1) and (
        compact_raw is None
        or (isinstance(compact_raw, str) and compact_raw.strip() == "")
    ):
        compact = True

    # Compact mode: return only path and line range
    if os.environ.get("DEBUG_REPO_SEARCH"):
        try:
            print("DEBUG_REPO_SEARCH: results=", len(results))
            for i, r in enumerate(results[:5]):
                print(f"  {i+1}: path={r.get('path')} symbol={r.get('symbol')} range={r.get('start_line')}-{r.get('end_line')}")
        except Exception:
            pass

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
            # Echo the user-provided compact flag in args, normalized via _to_bool to respect strings like "false"/"0"
            "compact": (_to_bool(compact_raw, compact)),
        },
        "used_rerank": bool(used_rerank),
        "rerank_counters": rerank_counters,
        "total": len(results),
        "results": results,
        **res,
    }


@mcp.tool()
async def repo_search_compat(**arguments) -> Dict[str, Any]:
    """Compatibility wrapper for repo_search.

    Accepts loose client payloads (queries/q/text/top_k and friends), normalizes
    them, and forwards to repo_search. Prevents -32602 from clients that send
    unexpected keys by only exposing **arguments here.
    """
    try:
        args = arguments or {}
        # Core query: prefer explicit query, else q/text; allow queries list passthrough
        query = args.get("query") or args.get("q") or args.get("text")
        queries = args.get("queries")
        # top_k alias for limit
        limit = args.get("limit")
        if (limit is None or (isinstance(limit, str) and str(limit).strip() == "")) and ("top_k" in args):
            limit = args.get("top_k")
        # not/ not_ normalization
        not_value = args.get("not_") if ("not_" in args) else args.get("not")

        # Build forward kwargs; pass alias keys too so repo_search's leniency picks them up
        forward = {
            "query": query,
            "limit": limit,
            "per_path": args.get("per_path"),
            "include_snippet": args.get("include_snippet"),
            "context_lines": args.get("context_lines"),
            "rerank_enabled": args.get("rerank_enabled"),
            "rerank_top_n": args.get("rerank_top_n"),
            "rerank_return_m": args.get("rerank_return_m"),
            "rerank_timeout_ms": args.get("rerank_timeout_ms"),
            "highlight_snippet": args.get("highlight_snippet"),
            "collection": args.get("collection"),
            "language": args.get("language"),
            "under": args.get("under"),
            "kind": args.get("kind"),
            "symbol": args.get("symbol"),
            "path_regex": args.get("path_regex"),
            "path_glob": args.get("path_glob"),
            "not_glob": args.get("not_glob"),
            "ext": args.get("ext"),
            "not_": not_value,
            "case": args.get("case"),
            "compact": args.get("compact"),
            # Alias passthroughs captured by repo_search(**kwargs)
            "queries": queries,
            "q": args.get("q"),
            "text": args.get("text"),
            "top_k": args.get("top_k"),
        }
        # Drop Nones to avoid overriding repo_search defaults unnecessarily
        clean = {k: v for k, v in forward.items() if v is not None}
        return await repo_search(**clean)
    except Exception as e:
        return {"error": f"repo_search_compat failed: {e}"}



@mcp.tool()
async def context_answer_compat(arguments: Any = None) -> Dict[str, Any]:
    """Compatibility wrapper for context_answer.

    Accepts a single 'arguments' dict like RMCP/HTTP clients send and forwards to
    context_answer with normalized keys. Mirrors repo_search_compat behavior but for
    the answer tool.
    """
    try:
        args = arguments or {}
        query = args.get("query") or args.get("q") or args.get("text")
        forward = {
            "query": query,
            "limit": args.get("limit"),
            "per_path": args.get("per_path"),
            "budget_tokens": args.get("budget_tokens"),
            "include_snippet": args.get("include_snippet"),
            "collection": args.get("collection"),
            "max_tokens": args.get("max_tokens"),
            "temperature": args.get("temperature"),
            "mode": args.get("mode"),
            "expand": args.get("expand"),
            # ---- Forward retrieval filters so router hints are honored ----
            "language": args.get("language"),
            "under": args.get("under"),
            "kind": args.get("kind"),
            "symbol": args.get("symbol"),
            "ext": args.get("ext"),
            "path_regex": args.get("path_regex"),
            "path_glob": args.get("path_glob"),
            "not_glob": args.get("not_glob"),
            "case": args.get("case"),
            # pass through NOT filter under either key
            "not_": args.get("not_") or args.get("not"),
        }
        clean = {k: v for k, v in forward.items() if v is not None}
        return await context_answer(**clean)
    except Exception as e:
        return {"error": f"context_answer_compat failed: {e}"}



@mcp.tool()
async def search_tests_for(
    query: Any = None,
    limit: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    under: Any = None,
    language: Any = None,
    compact: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    """Intent-specific wrapper to search for tests related to a query.
    Presets globs for common test locations and filenames across ecosystems.
    """
    globs = [
        "tests/**",
        "test/**",
        "**/*test*.*",
        "**/*_test.*",
        "**/Test*/**",
    ]
    # Allow caller to add more with path_glob kwarg
    extra_glob = kwargs.get("path_glob")
    if extra_glob:
        if isinstance(extra_glob, (list, tuple)):
            globs.extend([str(x) for x in extra_glob])
        else:
            globs.append(str(extra_glob))
    return await repo_search(
        query=query,
        limit=limit,
        include_snippet=include_snippet,
        context_lines=context_lines,
        under=under,
        language=language,
        path_glob=globs,
        compact=compact,
        **{k: v for k, v in kwargs.items() if k not in {"path_glob"}}
    )


@mcp.tool()
async def search_config_for(
    query: Any = None,
    limit: Any = None,
    include_snippet: Any = None,
    context_lines: Any = None,
    under: Any = None,
    compact: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    """Intent-specific wrapper to search likely configuration files for a service/query."""
    globs = [
        "**/*.yml",
        "**/*.yaml",
        "**/*.json",
        "**/*.toml",
        "**/*.ini",
        "**/*.env",
        "**/*.config",
        "**/*.conf",
        "**/*.properties",
        "**/*.csproj",
        "**/*.props",
        "**/*.targets",
        "**/*.xml",
        "**/appsettings*.json",
    ]
    extra_glob = kwargs.get("path_glob")
    if extra_glob:
        if isinstance(extra_glob, (list, tuple)):
            globs.extend([str(x) for x in extra_glob])
        else:
            globs.append(str(extra_glob))
    return await repo_search(
        query=query,
        limit=limit,
        include_snippet=include_snippet,
        context_lines=context_lines,
        under=under,
        path_glob=globs,
        compact=compact,
        **{k: v for k, v in kwargs.items() if k not in {"path_glob"}}
    )


@mcp.tool()
async def search_callers_for(
    query: Any = None,
    limit: Any = None,
    language: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    """Heuristic: find likely callers/usages of a symbol.
    Currently a thin wrapper over repo_search; future versions may expand using
    relation hints to prioritize files that reference the symbol.
    """
    return await repo_search(
        query=query,
        limit=limit,
        language=language,
        **kwargs,
    )


@mcp.tool()
async def search_importers_for(
    query: Any = None,
    limit: Any = None,
    language: Any = None,
    **kwargs,
) -> Dict[str, Any]:
    """Intent: find files likely importing/referencing a given module/symbol.
    Presets code-centric globs, accepts additional filters via kwargs.
    """
    globs = [
        "**/*.py", "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx", "**/*.mjs", "**/*.cjs",
        "**/*.go", "**/*.java", "**/*.cs", "**/*.rb", "**/*.php", "**/*.rs",
        "**/*.c", "**/*.h", "**/*.cpp", "**/*.hpp",
    ]
    extra_glob = kwargs.get("path_glob")
    if extra_glob:
        if isinstance(extra_glob, (list, tuple)):
            globs.extend([str(x) for x in extra_glob])
        else:
            globs.append(str(extra_glob))
    # Forward to repo_search with preset path_glob; caller can still pass other filters
    return await repo_search(
        query=query,
        limit=limit,
        language=language,
        path_glob=globs,
        **{k: v for k, v in kwargs.items() if k not in {"path_glob"}}
    )



@mcp.tool()
async def change_history_for_path(
    path: Any,
    collection: Any = None,
    max_points: Any = None,
) -> Dict[str, Any]:
    """Summarize recent changes for a file path using stored metadata.
    Returns counts, timestamps, churn, and distinct file_hashes. Best-effort.
    """
    p = str(path or "").strip()
    if not p:
        return {"error": "path required"}
    coll = str(collection or "").strip() or _default_collection()
    try:
        mcap = int(max_points) if max_points not in (None, "") else 200
    except Exception:
        mcap = 200
    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client import models as qmodels  # type: ignore
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY"),
            timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
        )
        filt = qmodels.Filter(
            must=[qmodels.FieldCondition(key="metadata.path", match=qmodels.MatchValue(value=p))]
        )
        page = None
        total = 0
        hashes = set()
        last_mods = []
        ingested = []
        churns = []
        while total < mcap:
            sc, page = await asyncio.to_thread(
                lambda: client.scroll(collection_name=coll, with_payload=True, with_vectors=False, limit=200, offset=page, scroll_filter=filt)
            )
            if not sc:
                break
            for pt in sc:
                md = (getattr(pt, "payload", {}) or {}).get("metadata") or {}
                fh = md.get("file_hash")
                if fh:
                    hashes.add(str(fh))
                lm = md.get("last_modified_at")
                ia = md.get("ingested_at")
                ch = md.get("churn_count")
                if lm is not None:
                    last_mods.append(int(lm))
                if ia is not None:
                    ingested.append(int(ia))
                if ch is not None:
                    churns.append(int(ch))
                total += 1
                if total >= mcap:
                    break
        summary = {
            "path": p,
            "points_scanned": total,
            "distinct_hashes": len(hashes),
            "last_modified_min": min(last_mods) if last_mods else None,
            "last_modified_max": max(last_mods) if last_mods else None,
            "ingested_min": min(ingested) if ingested else None,
            "ingested_max": max(ingested) if ingested else None,
            "churn_count_max": max(churns) if churns else None,
        }
        return {"ok": True, "summary": summary}
    except Exception as e:
        return {"ok": False, "error": str(e), "path": p}


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
    # Unwrap kwargs if MCP client sent everything in a single kwargs string
    if kwargs and not query and not limit:
        # If all named params are None and kwargs has content, assume wrapped call
        query = kwargs.get("query", query)
        limit = kwargs.get("limit", limit)
        per_path = kwargs.get("per_path", per_path)
        include_memories = kwargs.get("include_memories", include_memories)
        memory_weight = kwargs.get("memory_weight", memory_weight)
        per_source_limits = kwargs.get("per_source_limits", per_source_limits)
        include_snippet = kwargs.get("include_snippet", include_snippet)
        context_lines = kwargs.get("context_lines", context_lines)
        rerank_enabled = kwargs.get("rerank_enabled", rerank_enabled)
        rerank_top_n = kwargs.get("rerank_top_n", rerank_top_n)
        rerank_return_m = kwargs.get("rerank_return_m", rerank_return_m)
        rerank_timeout_ms = kwargs.get("rerank_timeout_ms", rerank_timeout_ms)
        highlight_snippet = kwargs.get("highlight_snippet", highlight_snippet)
        collection = kwargs.get("collection", collection)
        language = kwargs.get("language", language)
        under = kwargs.get("under", under)
        kind = kwargs.get("kind", kind)
        symbol = kwargs.get("symbol", symbol)
        path_regex = kwargs.get("path_regex", path_regex)
        path_glob = kwargs.get("path_glob", path_glob)
        not_glob = kwargs.get("not_glob", not_glob)
        ext = kwargs.get("ext", ext)
        not_ = kwargs.get("not_", not_)
        case = kwargs.get("case", case)
        compact = kwargs.get("compact", compact)

    # Normalize inputs
    coll = (collection or _default_collection()) or ""
    mcoll = (os.environ.get("MEMORY_COLLECTION_NAME") or coll) or ""
    use_sse_memory = str(os.environ.get("MEMORY_SSE_ENABLED", "false")).lower() in (
        "1",
        "true",
        "yes",
    )
    # Auto-detect memory collection if not explicitly set
    if include_memories and not os.environ.get("MEMORY_COLLECTION_NAME"):
        try:
            from qdrant_client import QdrantClient  # type: ignore

            # Optional: disable auto-detect and/or use cached result
            if str(os.environ.get("MEMORY_AUTODETECT", "1")).lower() not in (
                "1",
                "true",
                "yes",
                "on",
            ):
                raise RuntimeError("auto-detect disabled")
            import time

            ttl = float(os.environ.get("MEMORY_COLLECTION_TTL_SECS", "300") or 300)
            if (
                _MEM_COLL_CACHE["name"]
                and (time.time() - float(_MEM_COLL_CACHE["ts"] or 0.0)) < ttl
            ):
                mcoll = _MEM_COLL_CACHE["name"]
                raise RuntimeError("use cache")
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
            )
            info = await asyncio.to_thread(client.get_collections)
            best_name = None
            best_hits = -1
            for c in info.collections:
                name = getattr(c, "name", None)
                if not name:
                    continue
                # Sample a small page for memory-like payloads
                try:
                    pts, _ = await asyncio.to_thread(
                        lambda: client.scroll(
                            collection_name=name,
                            with_payload=True,
                            with_vectors=False,
                            limit=300,
                        )
                    )
                    hits = 0
                    for pt in pts:
                        pl = getattr(pt, "payload", {}) or {}
                        md = pl.get("metadata") or {}
                        path = md.get("path")
                        content = (
                            pl.get("content")
                            or pl.get("text")
                            or pl.get("information")
                            or md.get("information")
                        )
                        if not path and content:
                            hits += 1
                    if hits > best_hits:
                        best_hits = hits
                        best_name = name
                except Exception:
                    continue
            if best_name and best_hits > 0:
                mcoll = best_name
                try:
                    import time

                    _MEM_COLL_CACHE["name"] = best_name
                    _MEM_COLL_CACHE["ts"] = time.time()
                except Exception:
                    pass
        except Exception:
            pass

    try:
        lim = int(limit) if (limit is not None and str(limit).strip() != "") else 10
    except Exception:
        lim = 10
    try:
        per_path_val = (
            int(per_path)
            if (per_path is not None and str(per_path).strip() != "")
            else 1
        )
    except Exception:
        per_path_val = 1

    # Normalize queries to list (accept q/text aliases)
    queries: List[str] = []
    if query is None or (isinstance(query, str) and query.strip() == ""):
        q_alt = kwargs.get("q") or kwargs.get("text")
        if q_alt is not None:
            query = q_alt
    if isinstance(query, (list, tuple)):
        queries = [str(q).strip() for q in query if str(q).strip()]
    elif isinstance(query, str):
        queries = _to_str_list_relaxed(query)
    elif query is not None and str(query).strip() != "":
        queries = [str(query).strip()]

    # Accept common alias keys and camelCase from clients
    if (limit is None or (isinstance(limit, str) and limit.strip() == "")) and (
        "top_k" in kwargs
    ):
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
    if len(queries) > 1 and (
        compact_raw is None
        or (isinstance(compact_raw, str) and compact_raw.strip() == "")
    ):
        smart_compact = True
    # If snippets are requested, disable compact to preserve snippet field
    if include_snippet and str(include_snippet).lower() not in ("", "false", "0", "no"):
        smart_compact = False
        compact_raw = False
    eff_compact = (
        True if (smart_compact or (str(compact_raw).lower() == "true")) else False
    )

    # Per-source limits
    code_limit = lim
    mem_limit = 0
    include_mem = False
    if include_memories is not None and str(include_memories).lower() in (
        "true",
        "1",
        "yes",
    ):  # opt-in
        include_mem = True
        # Parse per_source_limits if provided; accept JSON-ish strings as well
        code_limit = lim
        mem_limit = min(3, lim)  # sensible default
        try:
            psl = per_source_limits
            # Some clients stringify payloads; parse if JSON-ish
            if isinstance(psl, str) and _looks_jsonish_string(psl):
                _ps = _maybe_parse_jsonish(psl)
                if isinstance(_ps, dict):
                    psl = _ps
            if isinstance(psl, dict):
                code_limit = int(psl.get("code", code_limit))
                mem_limit = int(psl.get("memory", mem_limit))
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
            # Best-effort: poll memory MCP /readyz on its health port to avoid init race
            try:
                from urllib.parse import urlparse
                import urllib.request, time

                ready_attempts = int(
                    os.environ.get("MEMORY_MCP_READY_RETRIES", "5") or 5
                )
                ready_backoff = float(
                    os.environ.get("MEMORY_MCP_READY_BACKOFF", "0.2") or 0.2
                )
                health_port = int(
                    os.environ.get("MEMORY_MCP_HEALTH_PORT", "18000") or 18000
                )
                pu = urlparse(base_url)
                host = pu.hostname or "mcp"
                scheme = pu.scheme or "http"
                readyz = f"{scheme}://{host}:{health_port}/readyz"

                def _poll_ready():
                    for i in range(max(1, ready_attempts)):
                        try:
                            with urllib.request.urlopen(readyz, timeout=1.5) as r:
                                if getattr(r, "status", 200) == 200:
                                    return True
                        except Exception:
                            time.sleep(ready_backoff * (i + 1))
                    return False

                try:
                    await asyncio.to_thread(_poll_ready)
                except Exception:
                    pass
            except Exception:
                pass

            async with Client(base_url) as c:
                tools = None
                attempts = int(os.environ.get("MEMORY_MCP_LIST_RETRIES", "3") or 3)
                backoff = float(os.environ.get("MEMORY_MCP_LIST_BACKOFF", "0.2") or 0.2)
                last_err = None
                for i in range(max(1, attempts)):
                    try:
                        tools = await asyncio.wait_for(c.list_tools(), timeout=timeout)
                        if tools:
                            break
                    except Exception as e:
                        last_err = e
                        try:
                            await asyncio.sleep(backoff * (i + 1))
                        except Exception:
                            pass
                if tools is None:
                    raise last_err or RuntimeError(
                        "list_tools failed before initialization"
                    )
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
                            res_obj = await asyncio.wait_for(
                                c.call_tool(tool_name, args), timeout=timeout
                            )
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
                        def push_text(
                            txt: str,
                            md: Dict[str, Any] | None = None,
                            score: float | int | None = None,
                        ):
                            if not txt:
                                return
                            mem_hits.append(
                                {
                                    "source": "memory",
                                    "score": float(score or 1.0),
                                    "content": txt,
                                    "metadata": (md or {}),
                                }
                            )

                        if isinstance(rd, dict):
                            cont = rd.get("content")
                            if isinstance(cont, list):
                                for c in cont:
                                    try:
                                        ctype = c.get("type")
                                        if ctype == "text" and isinstance(
                                            c.get("text"), str
                                        ):
                                            push_text(c["text"], {})
                                        elif ctype == "json":
                                            j = c.get("json")
                                            if isinstance(j, list):
                                                for it in j:
                                                    if isinstance(it, dict):
                                                        push_text(
                                                            str(
                                                                it.get("text")
                                                                or it.get("content")
                                                                or it.get("information")
                                                                or ""
                                                            ),
                                                            it.get("metadata") or {},
                                                            it.get("score") or 1.0,
                                                        )
                                            elif isinstance(j, dict):
                                                items = (
                                                    j.get("results")
                                                    or j.get("items")
                                                    or j.get("memories")
                                                    or j.get("data")
                                                )
                                                if isinstance(items, list):
                                                    for it in items:
                                                        if isinstance(it, dict):
                                                            push_text(
                                                                str(
                                                                    it.get("text")
                                                                    or it.get("content")
                                                                    or it.get(
                                                                        "information"
                                                                    )
                                                                    or ""
                                                                ),
                                                                it.get("metadata")
                                                                or {},
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
                                                str(
                                                    it.get("text")
                                                    or it.get("content")
                                                    or it.get("information")
                                                    or ""
                                                ),
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

            client = QdrantClient(
                url=QDRANT_URL,
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
            )
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            vec_name = sanitize_vector_name(model_name)
            model = _get_embedding_model(model_name)

            qtext = " ".join([q for q in queries if q]).strip() or queries[0]
            v = next(model.embed([qtext])).tolist()
            k = max(mem_limit, 5)
            res = await asyncio.to_thread(
                lambda: client.search(
                    collection_name=mcoll,
                    query_vector={"name": vec_name, "vector": v},
                    limit=k,
                    with_payload=True,
                )
            )
            for pt in res:
                payload = getattr(pt, "payload", {}) or {}
                md = payload.get("metadata") or {}
                path = str(md.get("path") or "")
                start_line = md.get("start_line")
                end_line = md.get("end_line")
                content = (
                    payload.get("content")
                    or payload.get("text")
                    or payload.get("information")
                    or md.get("information")
                )
                kind = (md.get("kind") or payload.get("kind") or "").lower()
                source_tag = (md.get("source") or payload.get("source") or "").lower()
                flagged = kind in (
                    "memory",
                    "preference",
                    "note",
                    "policy",
                    "infra",
                    "chat",
                ) or source_tag in ("memory", "chat")
                is_memory_like = (
                    (not path)
                    or (start_line in (None, 0) and end_line in (None, 0))
                    or flagged
                )
                if is_memory_like and content:
                    mem_hits.append(
                        {
                            "source": "memory",
                            "score": float(getattr(pt, "score", 0.0) or 0.0),
                            "content": content,
                            "metadata": md,
                        }
                    )
        except Exception:  # pragma: no cover
            pass

    # Fallback: lightweight substring scan over a capped scroll if vector name mismatch
    if include_mem and mem_limit > 0 and not mem_hits and queries:
        try:
            from qdrant_client import QdrantClient  # type: ignore

            client = QdrantClient(
                url=QDRANT_URL,
                api_key=os.environ.get("QDRANT_API_KEY"),
                timeout=float(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
            )
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
                    payload = getattr(pt, "payload", {}) or {}
                    md = payload.get("metadata") or {}
                    path = str(md.get("path") or "")
                    start_line = md.get("start_line")
                    end_line = md.get("end_line")
                    content = (
                        payload.get("content")
                        or payload.get("text")
                        or payload.get("information")
                        or md.get("information")
                    )
                    kind = (md.get("kind") or payload.get("kind") or "").lower()
                    source_tag = (
                        md.get("source") or payload.get("source") or ""
                    ).lower()
                    flagged = kind in (
                        "memory",
                        "preference",
                        "note",
                        "policy",
                        "infra",
                        "chat",
                    ) or source_tag in ("memory", "chat")
                    is_memory_like = (
                        (not path)
                        or (start_line in (None, 0) and end_line in (None, 0))
                        or flagged
                    )
                    if not (is_memory_like and content):
                        continue
                    low = str(content).lower()
                    if any(tok in low for tok in tokens):
                        mem_hits.append(
                            {
                                "source": "memory",
                                "score": 0.5,  # nominal score for substring match; blended via memory_weight
                                "content": content,
                                "metadata": md,
                            }
                        )
                        if len(mem_hits) >= mem_limit:
                            break
                checked += len(sc)
        except Exception:
            pass

    # Blend results
    try:
        mw = (
            float(memory_weight)
            if (memory_weight is not None and str(memory_weight).strip() != "")
            else 0.3
        )
    except Exception:
        mw = 0.3

    # Build per-source lists with adjusted scores
    code_scored = [{**h, "score": float(h.get("score", 0.0))} for h in code_hits]
    mem_scored = [{**h, "score": float(h.get("score", 0.0)) * mw} for h in mem_hits]

    # Enforce per-source limits before final slice so callers actually get memory hits
    if include_mem and mem_limit > 0:
        code_scored.sort(key=lambda x: -float(x.get("score", 0.0)))
        mem_scored.sort(key=lambda x: -float(x.get("score", 0.0)))
        m_keep = min(len(mem_scored), mem_limit, lim)
        sel_mem = mem_scored[:m_keep]
        c_keep = max(0, min(len(code_scored), code_limit, lim - m_keep))
        sel_code = code_scored[:c_keep]
        blended = sel_code + sel_mem
        blended.sort(
            key=lambda x: (
                -float(x.get("score", 0.0)),
                x.get("source", ""),
                str(x.get("path", "")),
            )
        )
        # No need to slice again; sel_code+sel_mem already <= lim
    else:
        blended = code_scored
        blended.sort(
            key=lambda x: (
                -float(x.get("score", 0.0)),
                x.get("source", ""),
                str(x.get("path", "")),
            )
        )
        blended = blended[:lim]

    # Compact shaping if requested
    if eff_compact:
        compacted: List[Dict[str, Any]] = []
        for b in blended:
            if b.get("source") == "code":
                compacted.append(
                    {
                        "source": "code",
                        "path": b.get("path"),
                        "start_line": b.get("start_line") or 0,
                        "end_line": b.get("end_line") or 0,
                    }
                )
            else:
                compacted.append(
                    {
                        "source": "memory",
                        "content": (b.get("content") or "")[:500],
                    }
                )
        ret = {"results": compacted, "total": len(compacted)}
        if memory_note:
            ret["memory_note"] = memory_note
        return ret

    ret = {"results": blended, "total": len(blended)}
    if memory_note:
        ret["memory_note"] = memory_note
    return ret
@mcp.tool()
async def expand_query(query: Any = None, max_new: Any = None) -> Dict[str, Any]:
    """LLM-assisted query expansion. Returns up to 2 alternates.
    Uses the local llama.cpp decoder when enabled; otherwise returns []."""
    try:
        qlist: list[str] = []
        if isinstance(query, (list, tuple)):
            qlist = [str(x) for x in query if str(x).strip()]
        elif query is not None:
            qlist = [str(query)] if str(query).strip() else []
        cap = 2
        if max_new not in (None, ""):
            try:
                cap = max(0, min(2, int(max_new)))
            except Exception:
                cap = 2
        from scripts.refrag_llamacpp import LlamaCppRefragClient, is_decoder_enabled  # type: ignore
        if not is_decoder_enabled():
            return {"alternates": [], "hint": "decoder disabled: set REFRAG_DECODER=1 and start llamacpp (LLAMACPP_URL)"}
        if not qlist:
            return {"alternates": []}
        prompt = (
            "You expand code search queries. Given short queries, propose up to 2 compact alternates.\n"
            "Return JSON array of strings only. No explanations.\n"
            f"Queries: {qlist}\n"
        )
        client = LlamaCppRefragClient()
        out = client.generate_with_soft_embeddings(
            prompt=prompt,
            max_tokens=int(os.environ.get("EXPAND_MAX_TOKENS", "64") or 64),
            temperature=float(os.environ.get("EXPAND_TEMPERATURE", "0.08") or 0.08),
            top_k=int(os.environ.get("EXPAND_TOP_K", "30") or 30),
            top_p=float(os.environ.get("EXPAND_TOP_P", "0.9") or 0.9),
            stop=["\n\n"],
        )
        import json as _json
        alts: list[str] = []
        try:
            parsed = _json.loads(out)
            if isinstance(parsed, list):
                for s in parsed:
                    if isinstance(s, str) and s and s not in qlist:
                        alts.append(s)
                        if len(alts) >= cap:
                            break
        except Exception:
            pass
        return {"alternates": alts}
    except Exception as e:
        return {"alternates": [], "error": str(e)}



# Lightweight cleanup to reduce repetition from small models
def _cleanup_answer(text: str, max_chars: int | None = None) -> str:
    try:
        import re
        t = (text or "").strip()
        if not t:
            return t
        # If model emitted 'insufficient context' anywhere, keep only what precedes it; if nothing precedes, return it
        low = t.lower()
        idx = low.find("insufficient context")
        if idx >= 0:
            prefix = t[:idx].strip()
            if prefix:
                t = prefix
            else:
                return "insufficient context"
        # Collapse excessive whitespace
        t = re.sub(r"\s+", " ", t)
        # Sentence-split and normalize
        sents = re.split(r"(?<=[.!?])\s+", t)
        out, seen = [], set()
        # Patterns of generic disclaimers we want to drop
        drop_substr = [
            "the provided code snippets only show",
            "without additional context",
            "i cannot provide a complete summary",
            "to understand",
        ]
        for s in sents:
            ss = s.strip()
            if not ss:
                continue
            base = re.sub(r"[.!?]+$", "", ss).strip().lower()
            # Skip disclaimers/filler
            if any(pat in base for pat in drop_substr):
                continue
            # Skip standalone 'insufficient context' (already handled above)
            if base == "insufficient context":
                continue
            # De-duplicate by normalized key
            key = base
            if key in seen:
                continue
            seen.add(key)
            out.append(ss)
        if not out:
            # Nothing useful; fall back to canonical insufficient message if hinted
            return "insufficient context" if "insufficient context" in low else t
        t2 = " ".join(out)
        # Optional final cap
        if max_chars and max_chars > 0 and len(t2) > max_chars:
            t2 = t2[: max(0, max_chars - 3) ] + "..."
        return t2
    except Exception:
        return text



@mcp.tool()
async def context_answer(
    query: Any = None,
    limit: Any = None,
    per_path: Any = None,
    budget_tokens: Any = None,
    include_snippet: Any = None,
    collection: Any = None,
    max_tokens: Any = None,
    temperature: Any = None,
    mode: Any = None,  # "stitch" (default) or "pack"
    expand: Any = None,  # whether to LLM-expand queries (up to 2 alternates)
    **kwargs,
) -> Dict[str, Any]:
    """Answer a question using gate-first retrieval (ReFRAG Option B), assembling
    context spans with citations and synthesizing an answer via llama.cpp.

    Behavior:
    - Runs hybrid retrieval with optional ReFRAG gate-first using MINI vectors.
    - Applies micro-span merge + token budgeting (REFRAG_MODE=1) in-process.
    - Builds citations (path + line range) from the budgeted spans.
    - Calls llama.cpp to synthesize an answer from the assembled context.

    Env knobs (read at call time):
    - REFRAG_MODE=1 to enable span budgeting
    - REFRAG_GATE_FIRST=1 to enable MINI-vector gating
    - REFRAG_CANDIDATES (default 200) to size the gated candidate set
    - MICRO_BUDGET_TOKENS (e.g., 1200) total token budget across spans
    - MICRO_OUT_MAX_SPANS (e.g., 8) max number of citation spans to return
    - LLAMACPP_URL (default http://localhost:8080)
    - REFRAG_DECODER=1 to enable llama.cpp calls
    """
    # Unwrap kwargs if MCP client sent everything in a single kwargs string
    if kwargs and not query:
        query = kwargs.get("query", query)
        limit = kwargs.get("limit", limit)
        per_path = kwargs.get("per_path", per_path)
        budget_tokens = kwargs.get("budget_tokens", budget_tokens)
        include_snippet = kwargs.get("include_snippet", include_snippet)
        collection = kwargs.get("collection", collection)
        max_tokens = kwargs.get("max_tokens", max_tokens)
        temperature = kwargs.get("temperature", temperature)
        mode = kwargs.get("mode", mode)
        expand = kwargs.get("expand", expand)

    # Normalize query to list[str]
    queries: list[str] = []
    if isinstance(query, (list, tuple)):
        queries = [str(q).strip() for q in query if str(q).strip()]
    elif isinstance(query, str):
        queries = _to_str_list_relaxed(query)
    elif query is not None:
        s = str(query).strip()
        if s:
            queries = [s]
    if not queries:
        return {"error": "query required"}

    # Effective limits
    # For Q&A, we want more results per file to get comprehensive context
    try:
        lim = int(limit) if (limit is not None and str(limit).strip() != "") else 15
    except Exception:
        lim = 15
    try:
        # Default per_path=5 for Q&A (vs 1 for search) to get multiple snippets from same file
        # This ensures we capture both definitions and usages
        ppath = int(per_path) if (per_path is not None and str(per_path).strip() != "") else 5
    except Exception:
        ppath = 5

    # For identifier-focused questions, allow more snippets per file to capture both definition and usage
    try:
        import re as _re
        _ids0 = _re.findall(r"\b([A-Z_][A-Z0-9_]{2,})\b", " ".join(queries))
        if _ids0:
            ppath = max(ppath, 5)
    except Exception:
        pass

    # Collection + model setup (reuse indexer defaults)
    coll = (collection or _default_collection()) or ""
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    model = _get_embedding_model(model_name)

    # Prepare environment toggles for ReFRAG gate-first and budgeting
    # Acquire lock to avoid cross-request env clobbering
    _ENV_LOCK.acquire()

    prev = {
        "REFRAG_MODE": os.environ.get("REFRAG_MODE"),
        "REFRAG_GATE_FIRST": os.environ.get("REFRAG_GATE_FIRST"),
        "REFRAG_CANDIDATES": os.environ.get("REFRAG_CANDIDATES"),
        "COLLECTION_NAME": os.environ.get("COLLECTION_NAME"),
        "MICRO_BUDGET_TOKENS": os.environ.get("MICRO_BUDGET_TOKENS"),
    }
    # Enable ReFRAG gate-first for context compression
    os.environ["REFRAG_MODE"] = "1"
    os.environ["REFRAG_GATE_FIRST"] = os.environ.get("REFRAG_GATE_FIRST", "1") or "1"
    os.environ["COLLECTION_NAME"] = coll
    if budget_tokens is not None and str(budget_tokens).strip() != "":
        os.environ["MICRO_BUDGET_TOKENS"] = str(budget_tokens)

    # Run in-process hybrid search to get structured items with span budgeting info
    from scripts.hybrid_search import run_hybrid_search  # type: ignore

    # Optionally expand queries via local decoder (tight cap) when requested
    queries = list(queries)
    # For LLM answering, default to include snippets so the model sees actual code
    try:
        if include_snippet in (None, ""):
            include_snippet = True
    except Exception:
        include_snippet = True
    try:
        do_expand = (
            (expand is True) or
            (str(expand).strip().lower() in {"1","true","yes","on"}) or
            (str(os.environ.get("HYBRID_EXPAND", "0")).strip().lower() in {"1","true","yes","on"})
        )
    except Exception:
        do_expand = False
    if do_expand:
        try:
            from scripts.refrag_llamacpp import LlamaCppRefragClient, is_decoder_enabled  # type: ignore
            if is_decoder_enabled():
                prompt = (
                    "You expand code search queries. Given one or more short queries, "
                    "propose up to 2 compact alternates. Return JSON array of strings only.\n"
                    f"Queries: {queries}\n"
                )
                client = LlamaCppRefragClient()
                # tight decoding for expansions
                out = client.generate_with_soft_embeddings(
                    prompt=prompt, max_tokens=int(os.environ.get("EXPAND_MAX_TOKENS", "64") or 64),
                    temperature=float(os.environ.get("EXPAND_TEMPERATURE", "0.08") or 0.08),
                    top_k=int(os.environ.get("EXPAND_TOP_K", "30") or 30),
                    top_p=float(os.environ.get("EXPAND_TOP_P", "0.9") or 0.9),
                    stop=["\n\n"]
                )
                import json as _json
                alts = []
                try:
                    parsed = _json.loads(out)
                    if isinstance(parsed, list):
                        for s in parsed:
                            if isinstance(s, str) and s and s not in queries:
                                alts.append(s)
                                if len(alts) >= 2:
                                    break
                except Exception:
                    pass
                if alts:
                    queries.extend(alts)
        except Exception:
            pass

    # Default exclusions to avoid noisy self-test and cache artifacts
    # Also filter out metadata/config files that pollute Q&A context
    # This significantly improves answer quality by focusing on implementation code
    user_not_glob = kwargs.get("not_glob")
    if isinstance(user_not_glob, str):
        user_not_glob = [user_not_glob]
    base_excludes = [
        ".selftest_repo/",
        ".pytest_cache/",
        ".codebase/",      # Indexer metadata (state.json, cache.json, etc.)
        ".kiro/",          # IDE configs
        "node_modules/",   # Dependencies
        ".git/",           # VCS internals
    ]
    # Add robust variants for absolute and recursive matching
    # Simplified to avoid over-filtering
    def _variants(p: str) -> list[str]:
        p = str(p).strip().strip()
        if not p:
            return []
        p = p.replace("\\", "/").lstrip("/")
        # Only use recursive glob pattern to catch all cases
        return [f"**/{p}**"]

    # Build defaults and conditional exclusions (skip if explicitly mentioned in query)
    default_not_glob = []
    for b in base_excludes:
        default_not_glob.extend(_variants(b))
    qtext = " ".join(queries).lower()
    def _mentions_any(keys: list[str]) -> bool:
        return any(k in qtext for k in keys)
    maybe_excludes = []
    if not _mentions_any([".env", "dotenv", "environment variable", "env var"]):
        maybe_excludes += [".env", ".env.*"]
    if not _mentions_any(["docker-compose", "compose"]):
        maybe_excludes += ["docker-compose*.yml", "docker-compose*.yaml", "compose*.yml", "compose*.yaml"]
    if not _mentions_any(["lock", "package-lock.json", "pnpm-lock", "yarn.lock", "poetry.lock", "cargo.lock", "go.sum", "composer.lock"]):
        maybe_excludes += [
            "*.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
            "poetry.lock", "Cargo.lock", "go.sum", "composer.lock"
        ]
    if not _mentions_any(["appsettings", "settings.json", "config"]):
        maybe_excludes += ["appsettings*.json"]
    # Apply variants for all conditional patterns
    for pat in maybe_excludes:
        default_not_glob.extend(_variants(pat))

    # Dedup while preserving order
    seen = set()
    eff_not_glob = []
    for g in default_not_glob + (user_not_glob or []):
        s = str(g).strip()
        if s and s not in seen:
            eff_not_glob.append(s)
            seen.add(s)

    items = []
    err = None
    try:
        # Respect 'collection' arg by passing it via env to hybrid_search
        coll = (collection or kwargs.get("collection") or os.environ.get("COLLECTION_NAME") or "").strip()
        if coll:
            os.environ["COLLECTION_NAME"] = coll

        # Debug: log the search parameters
        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            print(f"DEBUG SEARCH PARAMS: queries={queries}, limit={int(max(lim, 4))}, per_path={int(max(ppath, 0))}")
            print(f"DEBUG ENV: REFRAG_MODE={os.environ.get('REFRAG_MODE')}, COLLECTION_NAME={os.environ.get('COLLECTION_NAME')}")
            print(f"DEBUG FILTERS: not_glob={eff_not_glob}")


        # Initial search (tighten file/area when hinted and no explicit path_glob)
        def _to_glob_list(val: Any) -> list[str]:
            if not val:
                return []
            if isinstance(val, (list, tuple, set)):
                return [str(x).strip() for x in val if str(x).strip()]
            vs = str(val).strip()
            return [vs] if vs else []

        req_language = kwargs.get("language") or None
        eff_language = req_language or ("python" if ("router" in qtext and not req_language) else None)

        user_path_glob = _to_glob_list(kwargs.get("path_glob"))
        eff_path_glob: list[str] = list(user_path_glob)

        user_under = kwargs.get("under") or None
        override_under = None
        if isinstance(user_under, str):
            _uu = user_under.strip()
            if _uu:
                if "/" not in _uu:
                    eff_path_glob.append(f"**/{_uu}")
                else:
                    override_under = _uu if _uu.startswith("/") else f"/{_uu.lstrip('./')}"
        elif user_under:
            override_under = str(user_under)

        # Router-specific tightening when the word 'router' is present
        if ("router" in qtext):
            if not eff_path_glob:
                eff_path_glob = ["**/mcp_router.py", "**/*router*.py"]
            if not eff_language:
                eff_language = "python"

        # Normalize path_glob list -> None when empty
        if eff_path_glob:
            dedup_pg = []
            seen_pg = set()
            for pg in eff_path_glob:
                pg_str = str(pg).strip()
                if not pg_str or pg_str in seen_pg:
                    continue
                seen_pg.add(pg_str)
                dedup_pg.append(pg_str)
            eff_path_glob = dedup_pg
        else:
            eff_path_glob = None
        # Sanitize symbol: ignore file-like strings passed as symbol
        sym_arg = kwargs.get("symbol") or None
        try:
            if sym_arg and ("/" in str(sym_arg) or "." in str(sym_arg)):
                sym_arg = None
        except Exception:
            pass
        # Query sharpening: extract code identifiers and add targeted search terms
        try:
            qj = " ".join(queries)
            import re as _re

            primary = _primary_identifier_from_queries(queries)
            if primary and any(word in qj.lower() for word in ["what is", "how is", "used", "usage", "define"]):
                def _add_query(q: str):
                    qs = q.strip()
                    if qs and qs not in queries:
                        queries.append(qs)

                # Always keep the original user tokens; just append focused probes
                _add_query(primary)
                _add_query(f"{primary} =")
                func_name = primary.lower().split("_")[0]
                if func_name and len(func_name) > 2:
                    _add_query(f"def {func_name}(")
            else:
                # For other queries, add basename if we have path_glob
                if eff_path_glob:
                    def _basename(p: str) -> str:
                        s = str(p).replace("\\", "/").strip()
                        return s.split("/")[-1] if "/" in s else s

                    basenames = []
                    if isinstance(eff_path_glob, (list, tuple)):
                        basenames = [_basename(p) for p in eff_path_glob]
                    else:
                        basenames = [_basename(str(eff_path_glob))]
                    for bn in basenames:
                        if bn and bn not in queries:
                            queries.append(bn)
        except Exception:
            pass

        # Debug: log effective retrieval filters before search
        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            try:
                print(f"DEBUG FILTERS: language={eff_language}, override_under={override_under}, symbol={sym_arg}")
                print(f"DEBUG FILTERS: path_glob={eff_path_glob}")
            except Exception as _e:
                print(f"DEBUG FILTERS: print failed: {_e}")


        items = run_hybrid_search(
            queries=queries,
            limit=int(max(lim, 4)),  # fetch a few extra for budgeting
            per_path=int(max(ppath, 0)),
            language=eff_language,
            under=override_under or None,
            kind=kwargs.get("kind") or None,
            symbol=sym_arg,
            ext=kwargs.get("ext") or None,
            not_filter=kwargs.get("not_") or kwargs.get("not") or None,
            case=kwargs.get("case") or None,
            path_regex=kwargs.get("path_regex") or None,
            path_glob=eff_path_glob,
            not_glob=eff_not_glob,
            expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower() in {"1","true","yes","on"},
            model=model,
        )

        # Augment retrieval to capture usage sites of the identifier (e.g., default args, function defs)
        try:
            import re as _re
            qj2 = " ".join(queries)
            _ids = _re.findall(r"\b([A-Z_][A-Z0-9_]{2,})\b", qj2)
            _asked = _ids[0] if _ids else ""
            if _asked:
                _fname = _asked.lower().split("_")[0]
                _usage_qs = []
                if _fname and len(_fname) >= 2:
                    _usage_qs.append(f"def {_fname}(")
                _usage_qs.extend([
                    f"{_asked})", f"{_asked},", f"= {_asked}", f"{_asked} =",
                    f"{_asked} = int(os.environ.get", f"int(os.environ.get(\"{_asked}\""
                ])
                _usage_qs = [u for u in _usage_qs if u and u not in queries]
                if _usage_qs:
                    usage_items = run_hybrid_search(
                        queries=list(queries) + _usage_qs,
                        limit=int(max(lim, 30)),
                        per_path=int(max(ppath, 10)),
                        language=eff_language,
                        under=override_under or None,
                        kind=kwargs.get("kind") or None,
                        symbol=sym_arg,
                        ext=kwargs.get("ext") or None,
                        not_filter=kwargs.get("not_") or kwargs.get("not") or None,
                        case=kwargs.get("case") or None,
                        path_regex=kwargs.get("path_regex") or None,
                        path_glob=eff_path_glob,
                        not_glob=eff_not_glob,
                        expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower() in {"1","true","yes","on"},
                        model=model,
                    )
                    # Merge unique by (path, start_line, end_line)
                    def _ikey(it: Dict[str, Any]):
                        return (
                            str(it.get("path") or ""),
                            int(it.get("start_line") or 0),
                            int(it.get("end_line") or 0),
                        )
                    _seen = { _ikey(it) for it in items }
                    added = 0
                    for it in usage_items:
                        k = _ikey(it)
                        if k not in _seen:
                            items.append(it)
                            _seen.add(k)
                            added += 1
                    if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                        print(f"DEBUG USAGE_AUGMENT: asked={_asked}, added={added}, total_items={len(items)}")

                    # Additional targeted search for specific identifier patterns
                    if _asked and len(_asked) >= 3:
                        targeted_qs = [
                            f"{_asked} = int(os.environ.get",
                            f"def {_fname}(rank: int, k: int = {_asked})" if _fname else None
                        ]
                        targeted_qs = [q for q in targeted_qs if q]
                        if targeted_qs:
                            targeted_items = run_hybrid_search(
                                queries=targeted_qs,
                                limit=10,
                                per_path=5,
                                language=eff_language,
                                under=override_under or None,
                                kind=kwargs.get("kind") or None,
                                symbol=sym_arg,
                                ext=kwargs.get("ext") or None,
                                not_filter=kwargs.get("not_") or kwargs.get("not") or None,
                                case=kwargs.get("case") or None,
                                path_regex=kwargs.get("path_regex") or None,
                                path_glob=eff_path_glob,
                                not_glob=eff_not_glob,
                                expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower() in {"1","true","yes","on"},
                                model=model,
                            )
                            # Merge targeted results with higher priority
                            for it in targeted_items:
                                k = _ikey(it)
                                if k not in _seen:
                                    # Insert at beginning for higher priority
                                    items.insert(0, it)
                                    _seen.add(k)
                                    added += 1
                            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                                print(f"DEBUG TARGETED_SEARCH: found {len(targeted_items)} items, added {len([it for it in targeted_items if _ikey(it) not in _seen])}")
        except Exception as _e:
            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                print(f"DEBUG USAGE_AUGMENT failed: {_e}")

        # Debug: log top retrieval results before any post-filters/budgeting
        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            try:
                print(f"DEBUG RETRIEVAL: got {len(items)} items")
                for i, it in enumerate(items[:5], 1):
                    p = str(it.get("path") or "")
                    s = int(it.get("start_line") or 0)
                    e = int(it.get("end_line") or 0)
                    score = it.get("score", "N/A")
                    raw_score = it.get("raw_score", "N/A")
                    print(f"DEBUG RETRIEVAL ITEM {i}: {p}:{s}-{e} score={score} raw_score={raw_score}")
            except Exception as _e:
                print(f"DEBUG RETRIEVAL: print failed: {_e}")

        # Post-filter by language using path heuristics when language is provided
        if req_language:
            try:
                from scripts.hybrid_search import lang_matches_path as _lmp  # type: ignore
            except Exception:
                _lmp = None
            def _ok_lang(it: Dict[str, Any]) -> bool:
                p = str(it.get("path") or "")
                if callable(_lmp):
                    try:
                        return bool(_lmp(str(req_language), p))
                    except Exception:
                        pass
                # Fallback simple ext mapping
                ext = (p.rsplit(".", 1)[-1] if "." in p else "").lower()
                table = {
                    "python": ["py"],
                    "typescript": ["ts", "tsx"],
                    "javascript": ["js", "jsx", "mjs", "cjs"],
                    "go": ["go"],
                    "rust": ["rs"],
                    "java": ["java"],
                    "php": ["php"],
                }
                return ext in table.get(str(req_language).lower(), [])
            items = [it for it in items if _ok_lang(it)]

        # Debug: log search results
        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            print(f"DEBUG SEARCH RESULTS: found {len(items)} items")
            for i, item in enumerate(items[:3]):
                print(f"  Item {i+1}: {item.get('path')} lines {item.get('start_line')}-{item.get('end_line')}")

        # Fallback: only if no strict filters like language were provided
        if (not items) and (not req_language):
            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                print("DEBUG: 0 items after gate-first; retrying without gating")
            _prev_gate = os.environ.get("REFRAG_GATE_FIRST")
            os.environ["REFRAG_GATE_FIRST"] = "0"
            try:
                items = run_hybrid_search(
                    queries=queries,
                    limit=int(max(lim, 4)),
                    per_path=int(max(ppath, 0)),
                    language=eff_language,
                    under=override_under or None,
                    kind=kwargs.get("kind") or None,
                    symbol=sym_arg,
                    ext=kwargs.get("ext") or None,
                    not_filter=kwargs.get("not_") or kwargs.get("not") or None,
                    case=kwargs.get("case") or None,
                    path_regex=kwargs.get("path_regex") or None,
                    path_glob=eff_path_glob,
                    not_glob=eff_not_glob,
                    expand=str(os.environ.get("HYBRID_EXPAND", "1")).strip().lower() in {"1","true","yes","on"},
                    model=model,
                )
            finally:
                if _prev_gate is not None:
                    os.environ["REFRAG_GATE_FIRST"] = _prev_gate
                else:
                    os.environ.pop("REFRAG_GATE_FIRST", None)

        # Filter out memory-like items without a valid path to avoid empty citations
        items = [it for it in items if str(it.get("path") or "").strip()]

        # Apply ReFRAG span budgeting to compress context (33% compression target)
        from scripts.hybrid_search import _merge_and_budget_spans  # type: ignore
        try:
            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                print(f"DEBUG: Before span budgeting: {len(items)} items")
            budgeted = _merge_and_budget_spans(items)
            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                print(f"DEBUG: After span budgeting: {len(budgeted)} items")
            # Safety: if budgeting removed everything, fall back to raw items
            if not budgeted and items:
                if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                    print("DEBUG: Span budgeting returned empty, using raw items")
                budgeted = items
        except Exception as e:
            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                print(f"DEBUG: Span budgeting failed: {e}, using raw items")
            budgeted = items

        # Enforce an output max spans knob - do this BEFORE env restore
        # Increased default from 8 to 12 for better context coverage
        try:
            out_max = int(os.environ.get("MICRO_OUT_MAX_SPANS", "12") or 12)
        except Exception:
            out_max = 12
        # Respect caller-provided limit, allowing limit=0 to suppress snippets entirely
        span_cap = max(0, min(out_max, max(0, int(lim))))
        source_spans = list(budgeted) if budgeted else list(items)

        # Prefer spans that actually contain the main identifier when one is present
        def _read_span_snippet(span: Dict[str, Any]) -> str:
            cached = span.get("_ident_snippet")
            if cached is not None:
                return str(cached)
            if not include_snippet:
                return ""
            try:
                path = str(span.get("path") or "")
                sline = int(span.get("start_line") or 0)
                eline = int(span.get("end_line") or 0)
                if not path or sline <= 0:
                    span["_ident_snippet"] = ""
                    return ""
                fp = path
                if not os.path.isabs(fp):
                    fp = os.path.join("/work", fp)
                realp = os.path.realpath(fp)
                if not realp.startswith("/work/"):
                    span["_ident_snippet"] = ""
                    return ""
                with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                si = max(1, sline - 1)
                ei = min(len(lines), max(sline, eline) + 1)
                snippet = "".join(lines[si - 1 : ei])
                span["_ident_snippet"] = snippet
                return snippet
            except Exception:
                span["_ident_snippet"] = ""
                return ""

        def _span_haystack(span: Dict[str, Any]) -> str:
            parts = [
                str(span.get("text") or ""),
                str(span.get("symbol") or ""),
                str((span.get("relations") or {}).get("symbol_path") if isinstance(span.get("relations"), dict) else ""),
                str(span.get("path") or ""),
                str(span.get("_ident_snippet") or ""),
            ]
            return " ".join(parts).lower()

        def _span_key(span: Dict[str, Any]) -> tuple[str, int, int]:
            return (
                str(span.get("path") or ""),
                int(span.get("start_line") or 0),
                int(span.get("end_line") or 0),
            )

        primary_ident = _primary_identifier_from_queries(queries)
        if primary_ident and source_spans:
            ident_lower = primary_ident.lower()
            spans_with_ident: list[Dict[str, Any]] = []
            spans_without_ident: list[Dict[str, Any]] = []

            for span in source_spans:
                hay = _span_haystack(span)
                contains = ident_lower in hay
                if not contains:
                    extra = _read_span_snippet(span)
                    if extra:
                        hay = (hay + " " + extra.lower()).strip()
                        contains = ident_lower in hay
                if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                    print(f"DEBUG IDENT HAY path={span.get('path')} contains_ident={'yes' if contains else 'no'} preview={hay[:80]}")
                if contains:
                    spans_with_ident.append(span)
                else:
                    spans_without_ident.append(span)
            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                print(f"DEBUG IDENT FILTER: ident={primary_ident}, with={len(spans_with_ident)}, without={len(spans_without_ident)}")
            if spans_with_ident:
                source_spans = spans_with_ident + spans_without_ident
            elif budgeted and items:
                # Try to pull identifier-bearing spans from the broader pool
                ident_candidates: list[Dict[str, Any]] = []
                seen = set()
                for span in items:
                    key = _span_key(span)
                    if key in seen:
                        continue
                    if ident_lower in _span_haystack(span):
                        ident_candidates.append(span)
                        seen.add(key)
                if ident_candidates:
                    for span in source_spans:
                        key = _span_key(span)
                        if key not in seen:
                            ident_candidates.append(span)
                            seen.add(key)
                    if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                        print(f"DEBUG IDENT AUGMENT: pulled {len(ident_candidates)} candidate spans for {primary_ident}")
                    source_spans = ident_candidates

        if span_cap:
            spans = source_spans[:span_cap]
        else:
            spans = []

        # Debug span selection
        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            print(f"DEBUG SPAN SELECTION: items={len(items)}, budgeted={len(budgeted)}, out_max={out_max}, lim={lim}, spans={len(spans)}")

    except Exception as e:
        err = str(e)
        spans = []
    finally:
        # Restore env to previous values to avoid cross-call bleed
        for k, v in prev.items():
            if v is None:
                try:
                    del os.environ[k]
                except Exception:
                    pass
            else:
                os.environ[k] = v
        # Release lock after environment restored
        _ENV_LOCK.release()

    if err is not None:
        return {"error": f"hybrid search failed: {err}", "citations": [], "query": queries}

    # Build citations and context payload for the decoder
    citations: list[Dict[str, Any]] = []
    context_blocks: list[str] = []
    # Prepare deterministic definition/usage extraction
    asked_ident = _primary_identifier_from_queries(queries)
    _def_line_exact: str = ""
    _def_id: int | None = None
    _usage_id: int | None = None
    for idx, it in enumerate(spans, 1):
        path = str(it.get("path") or "")
        sline = int(it.get("start_line") or 0)
        eline = int(it.get("end_line") or 0)
        _hostp = it.get("host_path")
        _contp = it.get("container_path")
        _cit = {
            "id": idx,
            "path": path,
            "start_line": sline,
            "end_line": eline,
        }
        if _hostp:
            _cit["host_path"] = _hostp
        if _contp:
            _cit["container_path"] = _contp
        citations.append(_cit)
        # For context_answer, always read from filesystem to get complete, untruncated content
        # (payload text may be truncated for storage efficiency)
        snippet = ""
        if path and sline and include_snippet:
            try:
                fp = path
                import os as _os
                if not _os.path.isabs(fp):
                    fp = _os.path.join("/work", fp)
                realp = _os.path.realpath(fp)
                # SECURITY: Verify the resolved path stays within /work to prevent path traversal
                if not realp.startswith("/work/"):
                    if _os.environ.get("DEBUG_CONTEXT_ANSWER"):
                        print(f"DEBUG: Blocked path traversal attempt: {path} -> {realp}")
                    snippet = ""
                else:
                    with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    try:
                        margin = int(_os.environ.get("CTX_READ_MARGIN", "1") or 1)
                    except Exception:
                        margin = 1
                    si = max(1, sline - margin)
                    ei = min(len(lines), max(sline, eline) + margin)
                    snippet = "".join(lines[si-1:ei])
            except Exception:
                snippet = snippet or ""
        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            print(f"DEBUG: Snippet {idx} {('(payload)' if it.get('text') else '(fs)')} {path}:{sline}-{eline}, length={len(snippet)}")
            if snippet:
                print(f"DEBUG: Snippet {idx} content preview: {snippet[:200]}")
                # Show full content for RRF_K line
                if "RRF_K" in snippet and len(snippet) < 500:
                    print(f"DEBUG: Snippet {idx} FULL RRF_K CONTENT: {repr(snippet)}")
            else:
                print(f"DEBUG: Snippet {idx} is EMPTY!")
        header = f"[{idx}] {path}:{sline}-{eline}"
        # Increased snippet size for better context (was 600, now 1200)
        # This provides ~10-15 lines of code per snippet for better LLM understanding
        try:
            MAX_SNIPPET_CHARS = int(os.environ.get("CTX_SNIPPET_CHARS", "1200") or 1200)
        except Exception:
            MAX_SNIPPET_CHARS = 1200
        if snippet and len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS] + "\n..."
        block = header + "\n" + (snippet.strip() if snippet else "(no code)")
        context_blocks.append(block)
        # Extract definition/usage occurrences for robust formatting
        try:
            if asked_ident and snippet:
                import re as _re
                for _ln in str(snippet).splitlines():
                    if not _def_line_exact and _re.match(rf"\s*{_re.escape(asked_ident)}\s*=", _ln):
                        _def_line_exact = _ln.strip()
                        _def_id = idx
                    elif (asked_ident in _ln) and (_def_id != idx):
                        if _usage_id is None:
                            _usage_id = idx
        except Exception:
            pass



    # Debug: log span details
    if os.environ.get("DEBUG_CONTEXT_ANSWER"):
        print(f"DEBUG: spans={len(spans)}, context_blocks={len(context_blocks)}")
        for i, block in enumerate(context_blocks[:3], 1):
            print(f"DEBUG: Context block {i}:")
            print(block[:300])
            print("---")



    # Stop sequences for Granite-4.0-Micro + optional env overrides
    stop_env = os.environ.get("DECODER_STOP", "")
    default_stops = ["<|end_of_text|>", "<|start_of_role|>"]  # Granite format tokens
    stops = default_stops + [s for s in (stop_env.split(",") if stop_env else []) if s]

    # Decoder parameter tuning for small models (defaults can be overridden via args or env)
    def _to_int(v, d):
        try:
            return int(v)
        except Exception:
            return d
    def _to_float(v, d):
        try:
            return float(v)
        except Exception:
            return d
    # Generation params optimized for (code-focused, deterministic)
    mtok = _to_int(max_tokens, _to_int(os.environ.get("DECODER_MAX_TOKENS", "200"), 200))  # Shorter for conciseness
    # Granite 4.0 models work best with temperature 0 for deterministic extraction
    temp = _to_float(temperature, _to_float(os.environ.get("DECODER_TEMPERATURE", "0"), 0.0))
    top_k = _to_int(os.environ.get("DECODER_TOP_K", "20"), 20)  # More focused
    top_p = _to_float(os.environ.get("DECODER_TOP_P", "0.85"), 0.85)  # More deterministic

    # Call llama.cpp decoder (requires REFRAG_DECODER=1)
    try:
        from scripts.refrag_llamacpp import LlamaCppRefragClient, is_decoder_enabled  # type: ignore
        if not is_decoder_enabled():
            return {
                "error": "decoder disabled: set REFRAG_DECODER=1 and start llamacpp",
                "citations": citations,
                "query": queries,
            }
        client = LlamaCppRefragClient()

        # SIMPLE APPROACH: One LLM call with all context, tight prompt, 300 token limit
        qtxt = "\n".join(queries)
        all_context = "\n\n".join(context_blocks) if context_blocks else "(no code found)"

        # Derive lightweight usage hint heuristics to anchor tiny models
        extra_hint = ""
        try:
            if ("def rrf(" in all_context) and ("/(k + rank)" in all_context or "/ (k + rank)" in all_context):
                extra_hint = "RRF (Reciprocal Rank Fusion) formula 1.0 / (k + rank); parameter k defaults to RRF_K in def rrf."
        except Exception:
            extra_hint = ""

        # Build a sources footer (IDs and paths) to guide the model and satisfy downstream consumers
        sources_footer = "\n".join([f"[{c.get('id')}] {c.get('path')}" for c in citations]) if citations else ""

        # Extract key identifiers from code to help tiny models stay grounded
        import re
        key_terms = set()
        for block in context_blocks:
            # Extract function/class names, constants, and variables
            # Match: def func_name, class ClassName, CONSTANT_NAME = value
            matches = re.findall(r'\b(?:def|class|const|let|var|function)\s+([A-Za-z_][A-Za-z0-9_]*)', block)
            matches += re.findall(r'\b([A-Z_]{2,})\s*=', block)  # CONSTANTS
            key_terms.update(matches[:10])  # Limit to avoid prompt bloat

        key_terms_str = ", ".join(sorted(key_terms)[:15]) if key_terms else "none found"

        # Use Granite's proven RAG pattern for strict document grounding
        # Based on official Granite 4.0 RAG examples that enforce "strictly aligning with facts"
        system_msg = (
            "You are a helpful assistant with access to the following code snippets. "
            "You may use one or more snippets to assist with the user query.\n\n"
            "You are given code snippets with [ID] path:lines format:\n"
            f"{all_context}\n\n"
            + (f"Key identifiers: {key_terms_str}\n" if key_terms_str != "none found" else "")
            + (f"Hint: {extra_hint}\n\n" if extra_hint else "")
            + "Write the response to the user's input by strictly aligning with the facts in the provided code snippets. "
            "Always answer in exactly two lines with citations: "
            "Definition: \"<exact code definition line(s) verbatim>\" [ID]\n"
            "Usage: <one-sentence summary of how/where it is used based only on the snippets> [ID]. "
            "If the definition or usage is not present in the snippets, state 'Not found in provided snippets.' for that line."
        )
        if sources_footer:
            system_msg += f"\nSources:\n{sources_footer}"

        user_msg = f"{qtxt}"

        # Use Granite-4.0-Micro chat template format
        # Based on official HF documentation: <|start_of_role|>role<|end_of_role|>content<|end_of_text|>
        prompt = (
            f"<|start_of_role|>system<|end_of_role|>{system_msg}<|end_of_text|>\n"
            f"<|start_of_role|>user<|end_of_role|>{user_msg}<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>"
        )

        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            print(f"DEBUG: Single LLM call, prompt length={len(prompt)}")

        answer = client.generate_with_soft_embeddings(
            prompt=prompt,
            max_tokens=mtok,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            stop=stops,
            repeat_penalty=float(os.environ.get("DECODER_REPEAT_PENALTY", "1.15") or 1.15),
            repeat_last_n=int(os.environ.get("DECODER_REPEAT_LAST_N", "128") or 128),
        )

        # Optional length cap: if CTX_SUMMARY_CHARS is a positive int, apply after cleanup; otherwise don't cap
        try:
            _cap_env = str(os.environ.get("CTX_SUMMARY_CHARS", "")).strip()
            _cap = int(_cap_env) if _cap_env not in {"", None} else 0
        except Exception:
            _cap = 0

        # Cleanup repetition and optionally cap length
        answer = _cleanup_answer(answer, max_chars=(_cap if _cap and _cap > 0 else None))

        # Enforce strict two-line output with grounded definition using extracted snippet, for production reliability
        try:
            import re as _re
            txt = (answer or "").strip()
            # Split on explicit markers if present; otherwise infer
            def_part = ""
            usage_part = ""
            if "Usage:" in txt:
                parts = txt.split("Usage:", 1)
                def_part = parts[0]
                usage_part = parts[1]
                if "Definition:" in def_part:
                    def_part = def_part.split("Definition:", 1)[1]
            elif "Definition:" in txt:
                def_part = txt.split("Definition:", 1)[1]
                usage_part = ""
            else:
                def_part = txt
                usage_part = ""

            # Build Definition line
            def_line = None
            def _fmt_citation(cid: int | None) -> str:
                return f" [{cid}]" if cid is not None else ""

            if asked_ident and _def_line_exact:
                cid = _def_id if (_def_id is not None) else (citations[0]["id"] if citations else None)
                def_line = f'Definition: "{_def_line_exact}"{_fmt_citation(cid)}'
            else:
                # Try to salvage from model output if it quoted something containing the asked identifier
                cand = def_part.strip().strip("\n ")
                if asked_ident and asked_ident not in cand:
                    cand = ""
                # Try to extract a quoted string
                m = _re.search(r'"([^"]+)"', cand)
                q = m.group(1) if m else cand
                if asked_ident and asked_ident in q:
                    cid = citations[0]["id"] if citations else None
                    def_line = f'Definition: "{q.strip()}"{_fmt_citation(cid)}'
            if not def_line:
                def_line = "Definition: Not found in provided snippets."

            # Build Usage line
            usage_text = usage_part.strip().replace("\n", " ") if usage_part else ""
            # Truncate excessive spaces
            usage_text = _re.sub(r"\s+", " ", usage_text).strip()
            if not usage_text:
                # Minimal grounded usage if we saw other occurrences
                if _usage_id is not None:
                    usage_text = "Appears in the shown code."  # keep generic but grounded
                elif extra_hint:
                    usage_text = extra_hint
                else:
                    usage_text = "Not found in provided snippets."
            # Attach a citation id if missing
            if "[" not in usage_text and "]" not in usage_text:
                uid = (
                    _usage_id
                    if (_usage_id is not None)
                    else (_def_id if (_def_id is not None) else (citations[0]["id"] if citations else None))
                )
                usage_line = f"Usage: {usage_text}{_fmt_citation(uid)}"
            else:
                usage_line = f"Usage: {usage_text}"

            # Final strict two-line output
            answer = f"{def_line}\n{usage_line}".strip()
        except Exception:
            # If anything goes wrong, keep the original cleaned answer
            answer = answer.strip()

        # Debug: log the cleaned/formatted LLM response
        if os.environ.get("DEBUG_CONTEXT_ANSWER"):
            print(f"DEBUG: cleaned+formatted LLM answer: '{answer}' (len={len(answer)})")

        # Simple hallucination detection for tiny models
        # Check if answer contains generic phrases that suggest it's not grounded in the code
        hallucination_phrases = [
            "in general", "typically", "usually", "commonly", "often",
            "best practice", "it depends", "various ways", "multiple approaches",
            "can be implemented", "there are several", "one way to"
        ]
        answer_lower = answer.lower()
        hallucination_score = sum(1 for phrase in hallucination_phrases if phrase in answer_lower)

        # If answer seems generic and doesn't reference citations, add a warning
        has_citation_refs = any(f"[{i}]" in answer for i in range(1, len(citations) + 1))
        if hallucination_score >= 2 and not has_citation_refs:
            if os.environ.get("DEBUG_CONTEXT_ANSWER"):
                print(f"DEBUG: Possible hallucination detected (score={hallucination_score}, has_refs={has_citation_refs})")
            # Prepend a disclaimer for low-confidence answers
            answer = f"[Low confidence - answer may be generic] {answer}"

    except Exception as e:
        return {"error": f"decoder call failed: {e}", "citations": citations, "query": queries}

    return {
        "answer": answer.strip(),
        "citations": citations,
        "query": queries,
        "used": {"gate_first": True, "refrag": True},
    }


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
        if str(os.environ.get("EMBEDDING_WARMUP", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _ = _get_embedding_model(
                os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            )
    except Exception:
        pass
    try:
        if str(os.environ.get("RERANK_WARMUP", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        } and str(os.environ.get("RERANKER_ENABLED", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            if str(os.environ.get("RERANK_IN_PROCESS", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                try:
                    from scripts.rerank_local import _get_rerank_session  # type: ignore

                    _ = _get_rerank_session()
                except Exception:
                    pass
            else:
                # Fire a tiny warmup rerank once via subprocess; ignore failures
                _env = os.environ.copy()
                _env["QDRANT_URL"] = QDRANT_URL
                _env["COLLECTION_NAME"] = _default_collection()
                _cmd = [
                    "python",
                    "/work/scripts/rerank_local.py",
                    "--query",
                    "warmup",
                    "--topk",
                    "3",
                    "--limit",
                    "1",
                ]
                subprocess.run(
                    _cmd, capture_output=True, text=True, env=_env, timeout=10
                )
    except Exception:
        pass

    # Start lightweight /readyz health endpoint in background (best-effort)
    try:
        _start_readyz_server()
    except Exception:
        pass

    transport = os.environ.get("FASTMCP_TRANSPORT", "sse").strip().lower()
    if transport == "stdio":
        # Run over stdio (for clients that don't support network transports)
        mcp.run(transport="stdio")
    elif transport in {"http", "streamable", "streamable_http", "streamable-http"}:
        # Streamable HTTP (recommended) — endpoint at /mcp (FastMCP default)
        try:
            mcp.settings.host = HOST
            mcp.settings.port = PORT
        except Exception:
            pass
        # Use the correct FastMCP transport name
        try:
            mcp.run(transport="streamable-http")
        except Exception:
            # Fallback to SSE only if HTTP truly unavailable
            mcp.settings.host = HOST
            mcp.settings.port = PORT
            mcp.run(transport="sse")
    else:
        # SSE (legacy) — endpoint at /sse
        mcp.settings.host = HOST
        mcp.settings.port = PORT
        mcp.run(transport="sse")
