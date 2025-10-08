#!/usr/bin/env python3
import os
import argparse
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")


def _sanitize_vector_name(model_name: str) -> str:
    name = model_name.strip().lower()
    if name in (
        "sentence-transformers/all-minilm-l6-v2",
        "sentence-transformers/all-minilm-l-6-v2",
        "sentence-transformers/all-minilm-l6-v2",
    ):
        return "fast-all-minilm-l6-v2"
    if "bge-base-en-v1.5" in name:
        return "fast-bge-base-en-v1.5"
    for ch in ["/", ".", " ", "_"]:
        name = name.replace(ch, "-")
    while "--" in name:
        name = name.replace("--", "-")
    return name


def rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def lexical_score(phrases: List[str], md: Dict[str, Any]) -> float:
    text = " ".join([
        str(md.get("path", "")),
        str(md.get("path_prefix", "")),
        str(md.get("symbol", "")),
        str(md.get("symbol_path", "")),
        str(md.get("code", ""))[:2000],
    ]).lower()
    s = 0.0
    for p in phrases:
        q = p.lower()
        if not q:
            continue
        if q in text:
            s += 1.0
    return s


def dense_query(client: QdrantClient, vec_name: str, v: List[float], flt, per_query: int) -> List[Any]:
    try:
        qp = client.query_points(
            collection_name=COLLECTION,
            query=v,
            using=vec_name,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=128),
            limit=per_query,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except Exception:
        res = client.search(
            collection_name=COLLECTION,
            query_vector={"name": vec_name, "vector": v},
            limit=per_query,
            with_payload=True,
            query_filter=flt,
        )
        return res


def main():
    ap = argparse.ArgumentParser(description="Hybrid search: dense + lexical RRF")
    ap.add_argument("--query", "-q", action="append", required=True, help="One or more query strings (multi-query)")
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--under", type=str, default=None)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--per-query", type=int, default=24)
    args = ap.parse_args()

    model = TextEmbedding(model_name=MODEL_NAME)
    vec_name = _sanitize_vector_name(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    # Build optional filter
    flt = None
    must = []
    if args.language:
        must.append(models.FieldCondition(key="metadata.language", match=models.MatchValue(value=args.language)))
    if args.under:
        must.append(models.FieldCondition(key="metadata.path_prefix", match=models.MatchValue(value=args.under)))
    flt = models.Filter(must=must) if must else None

    # Dense results per query
    embedded = [vec.tolist() for vec in model.embed(args.query)]
    result_sets: List[List[Any]] = [dense_query(client, vec_name, v, flt, args.per_query) for v in embedded]

    # RRF fusion
    score_map: Dict[str, Dict[str, Any]] = {}
    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(pid, {"pt": p, "s": 0.0})
            score_map[pid]["s"] += rrf(rank)

    # Lexical bump
    for pid, rec in list(score_map.items()):
        md = (rec["pt"].payload or {}).get("metadata") or {}
        rec["s"] += 0.25 * lexical_score(args.query, md)

    # Rank and print
    merged = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)[: args.limit]
    for m in merged:
        md = (m["pt"].payload or {}).get("metadata") or {}
        print(f"{m['s']:.3f}\t{md.get('path')}\t{md.get('symbol_path') or md.get('symbol') or ''}\t{md.get('start_line')}-{md.get('end_line')}")


if __name__ == "__main__":
    main()

