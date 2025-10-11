import os
import time
from typing import Any, Dict, Optional, List

from fastmcp import FastMCP
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

def _lex_hash_vector(text: str, dim: int = LEX_VECTOR_DIM) -> List[float]:
    v = [0.0] * dim
    for tok in (text or "").lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
        v[h] += 1.0
    # L2 normalize
    norm = sum(x*x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]

# Map model to named vector used in indexer

def _sanitize_vector_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    if "bge-base-en-v1.5" in name:
        return "fast-bge-base-en-v1.5"
    return name.replace("/", "-").replace("_", "-")[:64]

VECTOR_NAME = _sanitize_vector_name(EMBEDDING_MODEL)

mcp = FastMCP(name="memory-server")
HOST = os.environ.get("FASTMCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("FASTMCP_PORT", "8000") or 8000)

client = QdrantClient(url=QDRANT_URL, api_key=os.environ.get("QDRANT_API_KEY"))

# Ensure collection exists with dual vectors

def _ensure_collection(name: str):
    try:
        info = client.get_collection(name)
        return True
    except Exception:
        pass
    # Derive dense vector dimension from embedding model to avoid mismatch
    try:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
        _dense_dim = len(next(_model.embed(["__dim_probe__"])).tolist())
    except Exception:
        _dense_dim = 768
    vectors_cfg = {
        VECTOR_NAME: models.VectorParams(size=int(_dense_dim or 768), distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(size=LEX_VECTOR_DIM, distance=models.Distance.COSINE),
    }
    client.create_collection(collection_name=name, vectors_config=vectors_cfg)
    return True

_ensure_collection(DEFAULT_COLLECTION)

@mcp.tool
def store(
    information: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Store a memory entry into Qdrant (dual vectors consistent with indexer)."""
    coll = collection or DEFAULT_COLLECTION
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    dense = next(model.embed([str(information)])).tolist()
    lex = _lex_hash_vector(str(information))
    pid = int(time.time_ns() % (2**31 - 1))
    payload = {
        "information": str(information),
        "metadata": metadata or {"kind": "memory", "source": "memory"},
    }
    point = models.PointStruct(id=pid, vector={VECTOR_NAME: dense, LEX_VECTOR_NAME: lex}, payload=payload)
    client.upsert(collection_name=coll, points=[point], wait=True)
    return {"ok": True, "id": pid, "collection": coll, "vector": VECTOR_NAME}

@mcp.tool
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
    lex = _lex_hash_vector(str(query))

    # Harmonize alias: top_k -> limit
    lim = int(limit or top_k or 5)

    # Two searches then simple RRF-like merge
    res_dense = client.search(collection_name=coll, query_vector=(VECTOR_NAME, dense), limit=max(10, lim))
    res_lex = client.search(collection_name=coll, query_vector=(LEX_VECTOR_NAME, lex), limit=max(10, lim))

    def is_memory_like(payload: Dict[str, Any]) -> bool:
        md = (payload or {}).get("metadata") or {}
        path = md.get("path")
        kind = (md.get("kind") or "").lower()
        source = (md.get("source") or "").lower()
        return (not path) or (kind in {"memory", "preference", "note", "policy", "chat"}) or (source in {"memory", "chat"})

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
            scores[pid] = scores.get(pid, 0.0) + weight / (1.0 + getattr(r, "score", 0.0))
            items[pid] = {
                "id": getattr(r, "id", None),
                "score": getattr(r, "score", None),
                "information": pl.get("information") or pl.get("content") or pl.get("text"),
                "metadata": pl.get("metadata") or {},
            }

    add_hits(res_dense, 1.0)
    add_hits(res_lex, 0.9)

    ordered = sorted(items.values(), key=lambda x: scores.get(str(x["id"]), 0.0), reverse=True)[:limit]
    return {"ok": True, "results": ordered, "count": len(ordered)}

if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)

