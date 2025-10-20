import os
import time
from typing import Any, Dict, Optional, List
import json
import threading


from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient, models

# Env
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
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

mcp = FastMCP(name="memory-server")
HOST = os.environ.get("FASTMCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("FASTMCP_PORT", "8000") or 8000)

# Lightweight readiness endpoint on a separate health port (non-MCP), optional
try:
    HEALTH_PORT = int(os.environ.get("FASTMCP_HEALTH_PORT", "18000") or 18000)
except Exception:
    HEALTH_PORT = 18000


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
    try:
        info = client.get_collection(name)
        return True
    except Exception:
        pass
    # Derive dense vector dimension from embedding model to avoid mismatch
    # Derive dense vector dimension from embedding model to avoid mismatch
    try:
        _model_probe = TextEmbedding(model_name=EMBEDDING_MODEL)
        _dense_vec = next(_model_probe.embed(["probe"]))
        _dense_dim = len(getattr(_dense_vec, "tolist", lambda: _dense_vec)()) if hasattr(_dense_vec, "tolist") else len(_dense_vec)
    except Exception:
        try:
            _dense_dim = int(os.environ.get("EMBED_DIM", "768") or 768)
        except Exception:
            _dense_dim = 768
    vectors_cfg = {
        VECTOR_NAME: models.VectorParams(
            size=int(_dense_dim or 768), distance=models.Distance.COSINE
        ),
        LEX_VECTOR_NAME: models.VectorParams(
            size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    client.create_collection(collection_name=name, vectors_config=vectors_cfg)
    return True


_ensure_collection(DEFAULT_COLLECTION)


@mcp.tool()
def store(
    information: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Store a memory entry into Qdrant (dual vectors consistent with indexer)."""
    coll = collection or DEFAULT_COLLECTION
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    dense = next(model.embed([str(information)])).tolist()
    lex = _lex_hash_vector_text(str(information), LEX_VECTOR_DIM)
    pid = int(time.time_ns() % (2**31 - 1))
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
) -> Dict[str, Any]:
    """Find memory-like entries by vector similarity (dense + lexical fusion)."""
    coll = collection or DEFAULT_COLLECTION
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    dense = next(model.embed([str(query)])).tolist()
    lex = _lex_hash_vector_text(str(query), LEX_VECTOR_DIM)

    # Harmonize alias: top_k -> limit
    lim = int(limit or top_k or 5)

    # Two searches (prefer query_points) then simple RRF-like merge
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
