#!/usr/bin/env python3
import os
import argparse
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import re


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


RRF_K = int(os.environ.get("HYBRID_RRF_K", "60") or 60)
DENSE_WEIGHT = float(os.environ.get("HYBRID_DENSE_WEIGHT", "1.0") or 1.0)
LEXICAL_WEIGHT = float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "0.25") or 0.25)
EF_SEARCH = int(os.environ.get("QDRANT_EF_SEARCH", "128") or 128)
# Lightweight, configurable boosts
SYMBOL_BOOST = float(os.environ.get("HYBRID_SYMBOL_BOOST", "0.15") or 0.15)
RECENCY_WEIGHT = float(os.environ.get("HYBRID_RECENCY_WEIGHT", "0.1") or 0.1)
CORE_FILE_BOOST = float(os.environ.get("HYBRID_CORE_FILE_BOOST", "0.1") or 0.1)

# Minimal code-aware query expansion (quick win)
CODE_SYNONYMS = {
    "function": ["method", "def", "fn"],
    "class": ["type", "object"],
    "create": ["init", "initialize", "construct"],
    "get": ["fetch", "retrieve"],
    "set": ["assign", "update"],
}

def expand_queries(queries: List[str], language: str | None = None, max_extra: int = 2) -> List[str]:
    out: List[str] = list(queries)
    for q in list(queries):
        ql = q.lower()
        for word, syns in CODE_SYNONYMS.items():
            if word in ql:
                for s in syns[:max_extra]:
                    exp = re.sub(rf"\b{re.escape(word)}\b", s, q, flags=re.IGNORECASE)
                    if exp not in out:
                        out.append(exp)
    return out[: max(8, len(queries))]

def _env_truthy(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def rrf(rank: int, k: int = RRF_K) -> float:
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
            search_params=models.SearchParams(hnsw_ef=EF_SEARCH),
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
    ap.add_argument("--kind", type=str, default=None)
    ap.add_argument("--symbol", type=str, default=None)
    # Expansion enabled by default; allow disabling via --no-expand or HYBRID_EXPAND=0
    ap.add_argument("--expand", dest="expand", action="store_true", default=_env_truthy(os.environ.get("HYBRID_EXPAND"), True), help="Enable simple query expansion")
    ap.add_argument("--no-expand", dest="expand", action="store_false", help="Disable query expansion")
    # Per-path diversification enabled by default (1) unless overridden by env/flag
    ap.add_argument("--per-path", type=int, default=int(os.environ.get("HYBRID_PER_PATH", "1") or 1), help="Cap results per file path to diversify (0=off)")

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
    if args.kind:
        must.append(models.FieldCondition(key="metadata.kind", match=models.MatchValue(value=args.kind)))
    if args.symbol:
        must.append(models.FieldCondition(key="metadata.symbol", match=models.MatchValue(value=args.symbol)))
    flt = models.Filter(must=must) if must else None

    # Build query set (optionally expanded)
    queries = list(args.query)
    if args.expand:
        queries = expand_queries(queries, args.language)

    embedded = [vec.tolist() for vec in model.embed(queries)]
    result_sets: List[List[Any]] = [dense_query(client, vec_name, v, flt, args.per_query) for v in embedded]

    # RRF fusion (weighted)
    score_map: Dict[str, Dict[str, Any]] = {}
    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(pid, {"pt": p, "s": 0.0})
            score_map[pid]["s"] += DENSE_WEIGHT * rrf(rank)

    # Lexical bump + symbol boost; also collect recency
    timestamps: List[int] = []
    for pid, rec in list(score_map.items()):
        md = (rec["pt"].payload or {}).get("metadata") or {}
        rec["s"] += LEXICAL_WEIGHT * lexical_score(queries, md)
        ts = md.get("ingested_at")
        if isinstance(ts, int):
            timestamps.append(ts)
        sym_text = " ".join([str(md.get("symbol") or ""), str(md.get("symbol_path") or "")]).lower()
        for q in queries:
            ql = q.lower()
            if ql and ql in sym_text:
                rec["s"] += SYMBOL_BOOST
                break

    # Recency bump (normalize across results)
    if timestamps and RECENCY_WEIGHT > 0.0:
        tmin, tmax = min(timestamps), max(timestamps)
        span = max(1, tmax - tmin)
        for rec in score_map.values():
            md = (rec["pt"].payload or {}).get("metadata") or {}
            ts = md.get("ingested_at")
            if isinstance(ts, int):
                norm = (ts - tmin) / span
                rec["s"] += RECENCY_WEIGHT * norm

    # Rank
    ranked = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)

    # Optional diversification by path
    if args.per_path and args.per_path > 0:
        counts: Dict[str, int] = {}
        merged: List[Dict[str, Any]] = []
        for m in ranked:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path", ""))
            c = counts.get(path, 0)
            if c < args.per_path:
                merged.append(m)
                counts[path] = c + 1
            if len(merged) >= args.limit:
                break
    else:
        merged = ranked[: args.limit]

    for m in merged:
        md = (m["pt"].payload or {}).get("metadata") or {}
        print(f"{m['s']:.3f}\t{md.get('path')}\t{md.get('symbol_path') or md.get('symbol') or ''}\t{md.get('start_line')}-{md.get('end_line')}")


if __name__ == "__main__":
    main()

