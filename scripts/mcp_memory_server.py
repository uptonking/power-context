import os
from typing import Any, Dict, Optional, List
import json
import threading
from weakref import WeakKeyDictionary


# FastMCP server and request Context (ctx) for per-connection state
try:
    from mcp.server.fastmcp import FastMCP, Context  # type: ignore
except Exception:
    # Fallback: keep FastMCP import; treat Context as Any for type hints
    from mcp.server.fastmcp import FastMCP  # type: ignore
    Context = Any  # type: ignore

from qdrant_client import QdrantClient, models

# Env
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = (os.environ.get("DEFAULT_COLLECTION")
                      or os.environ.get("COLLECTION_NAME")
                      or "my-collection")
LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# Minimal embedding via fastembed (CPU)
from fastembed import TextEmbedding

# Simple hashing trick for lexical vector to match indexer
import hashlib


# Ensure repo roots are importable so 'scripts' resolves inside container
import sys as _sys
_roots_env = os.environ.get("WORK_ROOTS", "")
_roots = [p.strip() for p in _roots_env.split(",") if p.strip()] or ["/work", "/app"]
try:
    for _root in _roots:
        if _root and _root not in _sys.path:
            _sys.path.insert(0, _root)
except Exception:
    pass

# Map model to named vector used in indexer


# Use shared utils for consistent vector naming and lexical hashing
from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.utils import lex_hash_vector_text as _lex_hash_vector_text

VECTOR_NAME = _sanitize_vector_name(EMBEDDING_MODEL)

# I/O-safety knobs for memory server behavior
# These env vars allow tuning startup latency vs. first-call latency, especially important
# on slow storage backends (e.g., Ceph + HDD). See comments below for rationale.
MEMORY_ENSURE_ON_START = str(os.environ.get("MEMORY_ENSURE_ON_START", "1")).strip().lower() in {"1", "true", "yes", "on"}
MEMORY_COLD_SKIP_DENSE = str(os.environ.get("MEMORY_COLD_SKIP_DENSE", "0")).strip().lower() in {"1", "true", "yes", "on"}
MEMORY_PROBE_EMBED_DIM = str(os.environ.get("MEMORY_PROBE_EMBED_DIM", "1")).strip().lower() in {"1", "true", "yes", "on"}
try:
    MEMORY_VECTOR_DIM = int(os.environ.get("MEMORY_VECTOR_DIM") or os.environ.get("EMBED_DIM") or "768")
except Exception:
    MEMORY_VECTOR_DIM = 768

# Lazy embedding model cache with double-checked locking.
# RATIONALE: Avoid loading the embedding model (100–500 MB) on module import.
# On slow storage (Ceph + HDD), eager loading can cause 30–60s startup delays.
# Instead, load on first tool call (store/find). Subsequent calls reuse cached instance.
_EMBED_MODEL_CACHE: Dict[str, Any] = {}
_EMBED_MODEL_LOCK = threading.Lock()

def _get_embedding_model():
    """Lazily load and cache the embedding model to avoid startup I/O."""
    from fastembed import TextEmbedding
    m = _EMBED_MODEL_CACHE.get(EMBEDDING_MODEL)
    if m is None:
        with _EMBED_MODEL_LOCK:
            m = _EMBED_MODEL_CACHE.get(EMBEDDING_MODEL)
            if m is None:
                m = TextEmbedding(model_name=EMBEDDING_MODEL)
                _EMBED_MODEL_CACHE[EMBEDDING_MODEL] = m
    return m

# Track ensured collections to reduce redundant ensure calls.
# RATIONALE: Avoid repeated Qdrant network calls for the same collection.
_ENSURED = set()

def _ensure_once(name: str) -> bool:
    """Ensure collection exists, but only once per process (cached result)."""
    if name in _ENSURED:
        return True
    try:
        _ensure_collection(name)
        _ENSURED.add(name)
        return True
    except Exception:
        return False

mcp = FastMCP(name="memory-server")
HOST = os.environ.get("FASTMCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("FASTMCP_PORT", "8000") or 8000)

# Lightweight readiness endpoint on a separate health port (non-MCP), optional
try:
    HEALTH_PORT = int(os.environ.get("FASTMCP_HEALTH_PORT", "18000") or 18000)
except Exception:
    HEALTH_PORT = 18000

# In-memory session defaults (legacy token-based)
_SESSION_LOCK = threading.Lock()
SESSION_DEFAULTS: Dict[str, Dict[str, Any]] = {}
# In-memory per-connection defaults keyed by ctx.session (no token required)
_SESSION_CTX_LOCK = threading.Lock()
SESSION_DEFAULTS_BY_SESSION: "WeakKeyDictionary[Any, Dict[str, Any]]" = WeakKeyDictionary()


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
                        payload = {"ok": True, "app": "memory-server"}
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
                return

        srv = HTTPServer((HOST, HEALTH_PORT), H)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        return True
    except Exception:
        return False


client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))

# Ensure collection exists with dual vectors


def _ensure_collection(name: str):
    """Create collection if missing.

    Default behavior mirrors the original implementation for PR compatibility:
    - Probe the embedding model to detect the dense vector dimension (MEMORY_PROBE_EMBED_DIM=1)
    - Eager ensure on startup (MEMORY_ENSURE_ON_START=1)

    For slow storage backends (e.g., Ceph + HDD), set the following in your env:
    - MEMORY_PROBE_EMBED_DIM=0  -> skip model probing; use MEMORY_VECTOR_DIM/EMBED_DIM
    - MEMORY_ENSURE_ON_START=0  -> ensure lazily on first tool call
    """
    try:
        client.get_collection(name)
        return True
    except Exception:
        pass

    # Choose dense dimension based on config: probe (default) vs env-configured
    if MEMORY_PROBE_EMBED_DIM:
        try:
            from fastembed import TextEmbedding
            _model_probe = TextEmbedding(model_name=EMBEDDING_MODEL)
            _dense_vec = next(_model_probe.embed(["probe"]))
            if hasattr(_dense_vec, "tolist"):
                dense_dim = len(_dense_vec.tolist())
            else:
                try:
                    dense_dim = len(_dense_vec)
                except Exception:
                    dense_dim = int(os.environ.get("MEMORY_VECTOR_DIM") or os.environ.get("EMBED_DIM") or "768")
        except Exception:
            # Fallback to env-configured dimension if probing fails
            try:
                dense_dim = int(os.environ.get("MEMORY_VECTOR_DIM") or os.environ.get("EMBED_DIM") or "768")
            except Exception:
                dense_dim = 768
    else:
        dense_dim = int(MEMORY_VECTOR_DIM or 768)

    vectors_cfg = {
        VECTOR_NAME: models.VectorParams(size=int(dense_dim or 768), distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(size=LEX_VECTOR_DIM, distance=models.Distance.COSINE),
    }
    client.create_collection(collection_name=name, vectors_config=vectors_cfg)
    return True


# Optional eager collection ensure on startup (enabled by default for backward compatibility).
# Set MEMORY_ENSURE_ON_START=0 to defer ensure to first tool call (recommended on slow storage).
if MEMORY_ENSURE_ON_START:
    try:
        _ensure_collection(DEFAULT_COLLECTION)
    except Exception:
        pass

@mcp.tool()
def set_session_defaults(
    collection: Optional[str] = None,
    session: Optional[str] = None,
    ctx: Context = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Set defaults (e.g., collection) for subsequent calls.

    Behavior:
    - If a request Context is provided (normal with FastMCP), store defaults per-connection
      so subsequent calls on the same MCP session automatically use them (no token needed).
    - Optionally, also supports a lightweight token for clients that prefer cross-connection reuse.

    Precedence everywhere: explicit collection > per-connection defaults > token defaults > env default.
    """
    try:
        _extra = kwargs or {}
        if isinstance(_extra, dict) and "kwargs" in _extra:
            inner = _extra.get("kwargs")
            if isinstance(inner, dict):
                _extra = inner
            elif isinstance(inner, str):
                try:
                    _extra = json.loads(inner)
                except Exception:
                    _extra = {}
        if (not collection) and isinstance(_extra, dict) and _extra.get("collection") is not None:
            collection = _extra.get("collection")
        if (not session) and isinstance(_extra, dict) and _extra.get("session") is not None:
            session = _extra.get("session")
    except Exception:
        pass

    # Prepare defaults payload
    defaults: Dict[str, Any] = {}
    if isinstance(collection, str) and collection.strip():
        defaults["collection"] = collection.strip()

    # Store per-connection (preferred, no token required)
    try:
        if ctx is not None and getattr(ctx, "session", None) is not None and defaults:
            with _SESSION_CTX_LOCK:
                existing = SESSION_DEFAULTS_BY_SESSION.get(ctx.session) or {}
                existing.update(defaults)
                SESSION_DEFAULTS_BY_SESSION[ctx.session] = existing
    except Exception:
        pass

    # Optional: also support legacy token
    sid = (str(session).strip() if session is not None else "") or None
    if not sid:
        import uuid as _uuid
        sid = _uuid.uuid4().hex[:12]
    try:
        if defaults:
            with _SESSION_LOCK:
                existing = SESSION_DEFAULTS.get(sid) or {}
                existing.update(defaults)
                SESSION_DEFAULTS[sid] = existing
    except Exception:
        pass

    return {
        "ok": True,
        "session": sid,
        "defaults": (SESSION_DEFAULTS.get(sid, {}) if sid else {}),
        "applied": ("connection" if (ctx is not None and getattr(ctx, "session", None) is not None) else "token"),
    }


@mcp.tool()
def store(
    information: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
    session: Optional[str] = None,
    ctx: Context = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Store a memory entry into Qdrant (dual vectors consistent with indexer).

    Note: First call may be slow (model loads on first use). Subsequent calls are fast.
    """
    # Leniency: absorb nested 'kwargs' JSON some MCP clients send (so callers can pass
    # collection inside a single kwargs payload)
    try:
        _extra = kwargs or {}
        if isinstance(_extra, dict) and "kwargs" in _extra:
            inner = _extra.get("kwargs")
            if isinstance(inner, dict):
                _extra = inner
            elif isinstance(inner, str):
                try:
                    _extra = json.loads(inner)
                except Exception:
                    _extra = {}
        if (not collection) and isinstance(_extra, dict) and _extra.get("collection") is not None:
            collection = _extra.get("collection")
    except Exception:
        pass

    # Apply session default if provided via kwargs
    sid = None
    try:
        _sx = kwargs or {}
        if isinstance(_sx, dict) and "kwargs" in _sx:
            inner = _sx.get("kwargs")
            if isinstance(inner, dict):
                _sx = inner
            elif isinstance(inner, str):
                try:
                    _sx = json.loads(inner)
                except Exception:
                    _sx = {}
        if isinstance(_sx, dict) and _sx.get("session") is not None:
            sid = str(_sx.get("session")).strip()
    except Exception:
        pass

    # Prefer explicit session param if provided
    try:
        if session is not None and str(session).strip():
            sid = str(session).strip()
    except Exception:
        pass


    coll = (collection or "").strip()

    # 1) Per-connection defaults via ctx (no token required)
    if (not coll) and ctx is not None and getattr(ctx, "session", None) is not None:
        try:
            with _SESSION_CTX_LOCK:
                _d2 = SESSION_DEFAULTS_BY_SESSION.get(ctx.session) or {}
                _sc2 = str((_d2.get("collection") or "")).strip()
                if _sc2:
                    coll = _sc2
        except Exception:
            pass

    # 2) Legacy token-based session defaults
    if (not coll) and sid:
        try:
            with _SESSION_LOCK:
                _d = SESSION_DEFAULTS.get(sid) or {}
                _sc = str((_d.get("collection") or "")).strip()
                if _sc:
                    coll = _sc
        except Exception:
            pass

    # 3) Environment fallback
    coll = coll or DEFAULT_COLLECTION

    _ensure_once(coll)  # Lazy: only ensures collection once per process
    model = _get_embedding_model()  # Lazy: loads model on first call, cached thereafter
    dense = next(model.embed([str(information)])).tolist()
    lex = _lex_hash_vector_text(str(information), LEX_VECTOR_DIM)
    # Use UUID to avoid point ID collisions under concurrent load
    import uuid
    pid = uuid.uuid4().hex
    payload = {
        "information": str(information),
        "metadata": metadata or {"kind": "memory", "source": "memory"},
    }
    point = models.PointStruct(
        id=pid, vector={VECTOR_NAME: dense, LEX_VECTOR_NAME: lex}, payload=payload
    )
    client.upsert(collection_name=coll, points=[point], wait=True)
    return {"ok": True, "id": pid, "collection": coll, "vector": VECTOR_NAME}


@mcp.tool()
def find(
    query: str,
    limit: int = 5,
    collection: Optional[str] = None,
    top_k: Optional[int] = None,
    session: Optional[str] = None,
    ctx: Context = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Find memory-like entries by vector similarity (dense + lexical fusion).

    Note: First call may be slow if dense embedding is used (model loads on first use).
    Set MEMORY_COLD_SKIP_DENSE=1 to skip dense on the very first query (falls back to
    lexical-only search). Default is 0 (always use dense) for backward compatibility.
    """
    # Leniency: absorb nested 'kwargs' JSON some MCP clients send
    try:
        _extra = kwargs or {}
        if isinstance(_extra, dict) and "kwargs" in _extra:
            inner = _extra.get("kwargs")
            if isinstance(inner, dict):
                _extra = inner
            elif isinstance(inner, str):
                try:
                    _extra = json.loads(inner)
                except Exception:
                    _extra = {}
        if (not collection) and isinstance(_extra, dict) and _extra.get("collection") is not None:
            collection = _extra.get("collection")
    except Exception:
        pass

    # Apply session default if provided via kwargs
    sid = None
    try:
        _sx = kwargs or {}
        if isinstance(_sx, dict) and "kwargs" in _sx:
            inner = _sx.get("kwargs")
            if isinstance(inner, dict):
                _sx = inner
            elif isinstance(inner, str):
                try:
                    _sx = json.loads(inner)
                except Exception:
                    _sx = {}
        if isinstance(_sx, dict) and _sx.get("session") is not None:
            sid = str(_sx.get("session")).strip()
    except Exception:
        pass

    # Prefer explicit session param if provided
    try:
        if session is not None and str(session).strip():
            sid = str(session).strip()
    except Exception:
        pass

    coll = (collection or "").strip()

    # 1) Per-connection defaults via ctx (no token required)
    if (not coll) and ctx is not None and getattr(ctx, "session", None) is not None:
        try:
            with _SESSION_CTX_LOCK:
                _d2 = SESSION_DEFAULTS_BY_SESSION.get(ctx.session) or {}
                _sc2 = str((_d2.get("collection") or "")).strip()
                if _sc2:
                    coll = _sc2
        except Exception:
            pass

    # 2) Legacy token-based session defaults
    if (not coll) and sid:
        try:
            with _SESSION_LOCK:
                _d = SESSION_DEFAULTS.get(sid) or {}
                _sc = str((_d.get("collection") or "")).strip()
                if _sc:
                    coll = _sc
        except Exception:
            pass

    # 3) Environment fallback
    coll = coll or DEFAULT_COLLECTION
    _ensure_once(coll)

    # Cold-start optimization: skip dense embedding on first query if model not yet loaded.
    # RATIONALE: On slow storage, loading the embedding model (100–500 MB) can delay
    # the first response by 5–10s. By skipping dense on cold start, we return results
    # faster using lexical search alone. Subsequent queries use dense (model is cached).
    # Set MEMORY_COLD_SKIP_DENSE=0 to disable this optimization.
    use_dense = True
    if MEMORY_COLD_SKIP_DENSE and EMBEDDING_MODEL not in _EMBED_MODEL_CACHE:
        use_dense = False
    if use_dense:
        model = _get_embedding_model()
        dense = next(model.embed([str(query)])).tolist()
    else:
        dense = None
    lex = _lex_hash_vector_text(str(query), LEX_VECTOR_DIM)

    # Harmonize alias: top_k -> limit
    lim = int(limit or top_k or 5)

    # Two searches (prefer query_points) then simple RRF-like merge
    if use_dense:
        try:
            qp_dense = client.query_points(
                collection_name=coll,
                query=dense,
                using=VECTOR_NAME,
                limit=max(10, lim),
                with_payload=True,
            )
            res_dense = getattr(qp_dense, "points", qp_dense)
        except AttributeError:
            res_dense = client.search(
                collection_name=coll,
                query_vector=(VECTOR_NAME, dense),
                limit=max(10, lim),
                with_payload=True,
            )
    else:
        res_dense = []

    try:
        qp_lex = client.query_points(
            collection_name=coll,
            query=lex,
            using=LEX_VECTOR_NAME,
            limit=max(10, lim),
            with_payload=True,
        )
        res_lex = getattr(qp_lex, "points", qp_lex)
    except AttributeError:
        res_lex = client.search(
            collection_name=coll,
            query_vector=(LEX_VECTOR_NAME, lex),
            limit=max(10, lim),
            with_payload=True,
        )

    def is_memory_like(payload: Dict[str, Any]) -> bool:
        md = (payload or {}).get("metadata") or {}
        path = md.get("path")
        kind = (md.get("kind") or "").lower()
        source = (md.get("source") or "").lower()
        return (
            (not path)
            or (kind in {"memory", "preference", "note", "policy", "chat"})
            or (source in {"memory", "chat"})
        )

    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}

    def add_hits(hits, weight: float):
        for r in hits:
            pid = str(getattr(r, "id", None))
            if not pid:
                continue
            pl = getattr(r, "payload", {}) or {}
            if not is_memory_like(pl):
                continue
            scores[pid] = scores.get(pid, 0.0) + weight / (
                1.0 + getattr(r, "score", 0.0)
            )
            items[pid] = {
                "id": getattr(r, "id", None),
                "score": getattr(r, "score", None),
                "information": pl.get("information")
                or pl.get("content")
                or pl.get("text"),
                "metadata": pl.get("metadata") or {},
            }

    add_hits(res_dense, 1.0)
    add_hits(res_lex, 0.9)

    ordered = sorted(
        items.values(), key=lambda x: scores.get(str(x["id"]), 0.0), reverse=True
    )[:limit]
    return {"ok": True, "results": ordered, "count": len(ordered)}


if __name__ == "__main__":
    transport = os.environ.get("FASTMCP_TRANSPORT", "sse").strip().lower()
    # Start lightweight /readyz health endpoint in background (best-effort)
    try:
        _start_readyz_server()
    except Exception:
        pass

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
