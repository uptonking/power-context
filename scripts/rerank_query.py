#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict
from typing import List, Dict, Any

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# Env configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")


def derive_vector_name(model_name: str) -> str:
    name = model_name.strip().lower()
    if "bge-base-en-v1.5" in name:
        return "fast-bge-base-en-v1.5"
    if "minilm" in name:
        return "fast-all-minilm-l6-v2"
    # fallback sanitize
    for ch in ["/", ".", " ", "_"]:
        name = name.replace(ch, "-")
    while "--" in name:
        name = name.replace("--", "-")
    return name


def score_candidate(base_score: float, count: int, md: Dict[str, Any], want_lang: str, want_prefix: str) -> float:
    s = base_score
    # Encourage consensus across multiple phrasings
    s += 0.02 * max(0, count - 1)
    # Boosts for metadata matches
    lang = (md or {}).get("language") or ""
    if want_lang and lang.lower() == want_lang.lower():
        s += 0.03
    prefix = (md or {}).get("path_prefix") or ""
    if want_prefix and str(prefix).startswith(want_prefix):
        s += 0.03
    return s


def main():
    parser = argparse.ArgumentParser(description="Multi-query re-ranker for Qdrant code search (no new deps)")
    parser.add_argument("--query", "-q", action="append", required=True, help="Query text; repeat flag to add variants")
    parser.add_argument("--limit", type=int, default=8, help="Final top-N to print after re-ranking")
    parser.add_argument("--per-query", type=int, default=12, help="How many results to pull per query before fusing")
    parser.add_argument("--language", type=str, default="", help="Preferred language (boost)")
    parser.add_argument("--under", type=str, default="", help="Preferred path_prefix (boost), e.g. /work/scripts")
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL)
    model = TextEmbedding(model_name=MODEL)
    vec_name = derive_vector_name(MODEL)

    cand: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = defaultdict(int)

    for q in args.query:
        v = next(model.embed([q]))
        # Optional server-side filter to reduce bandwidth
        flt = None
        if args.language or args.under:
            must = []
            if args.language:
                must.append(models.FieldCondition(key="metadata.language", match=models.MatchValue(value=args.language)))
            if args.under:
                must.append(models.FieldCondition(key="metadata.path_prefix", match=models.MatchValue(value=args.under)))
            flt = models.Filter(must=must)

        # Prefer modern query_points API with 'using' for named vector
        try:
            qp = client.query_points(
                collection_name=COLLECTION,
                query=v.tolist(),
                using=vec_name,
                query_filter=flt,
                search_params=models.SearchParams(hnsw_ef=128),
                limit=args.per_query,
                with_payload=True,
            )
            res_points = getattr(qp, "points", qp)
        except Exception:
            # Fallback to deprecated search API
            res_points = client.search(
                collection_name=COLLECTION,
                query_vector={"name": vec_name, "vector": v.tolist()},
                limit=args.per_query,
                with_payload=True,
                query_filter=flt,
            )
        for p in res_points:
            pid = str(p.id)
            counts[pid] += 1
            if pid not in cand or p.score > cand[pid]["base_score"]:
                cand[pid] = {
                    "base_score": float(p.score),
                    "payload": p.payload or {},
                }

    fused = []
    for pid, data in cand.items():
        md = (data["payload"] or {}).get("metadata") or {}
        final = score_candidate(data["base_score"], counts[pid], md, args.language, args.under)
        fused.append((final, pid, data))

    fused.sort(key=lambda x: x[0], reverse=True)

    print(f"Multi-query rerank: queries={len(args.query)} model={MODEL} vec={vec_name}")
    for i, (final, pid, data) in enumerate(fused[: args.limit], 1):
        md = (data["payload"] or {}).get("metadata") or {}
        info = (data["payload"] or {}).get("information") or (data["payload"] or {}).get("document")
        path = md.get("path")
        lang = md.get("language")
        print({
            "rank": i,
            "score": round(final, 4),
            "path": path,
            "language": lang,
            "information": info,
        })


if __name__ == "__main__":
    main()

