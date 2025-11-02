#!/usr/bin/env python3
import os
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
VEC_NAME = "fast-bge-base-en-v1.5"

client = QdrantClient(url=QDRANT_URL)
emb = TextEmbedding(model_name=MODEL)
q = "function that chunks code lines with overlap for semantic indexing"
vec = next(emb.embed([q]))
res = client.search(
    collection_name=COLLECTION,
    query_vector={"name": VEC_NAME, "vector": vec.tolist()},
    limit=5,
    with_payload=True,
)
for p in res:
    info = (p.payload or {}).get("information")
    md = (p.payload or {}).get("metadata") or {}
    print(
        {
            "score": round(p.score, 4),
            "information": info,
            "path": md.get("path"),
            "language": md.get("language"),
        }
    )
