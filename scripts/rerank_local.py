#!/usr/bin/env python3
import os
import argparse
from typing import List, Dict, Any

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# Optional imports for local ONNX reranker
try:
    import onnxruntime as ort  # type: ignore
    from tokenizers import Tokenizer  # type: ignore
except Exception:  # pragma: no cover
    ort = None
    Tokenizer = None

COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")

RERANKER_ONNX_PATH = os.environ.get("RERANKER_ONNX_PATH", "")
RERANKER_TOKENIZER_PATH = os.environ.get("RERANKER_TOKENIZER_PATH", "")


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


def dense_results(client: QdrantClient, vec_name: str, query: str, flt, topk: int) -> List[Any]:
    model = TextEmbedding(model_name=MODEL_NAME)
    vec = next(model.embed([query])).tolist()
    try:
        qp = client.query_points(
            collection_name=COLLECTION,
            query=vec,
            using=vec_name,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=128),
            limit=topk,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except Exception:
        res = client.search(
            collection_name=COLLECTION,
            query_vector={"name": vec_name, "vector": vec},
            limit=topk,
            with_payload=True,
            query_filter=flt,
        )
        return res


def prepare_pairs(query: str, points: List[Any]) -> List[str]:
    pairs = []
    for p in points:
        md = (p.payload or {}).get("metadata") or {}
        # prefer symbol_path or symbol + first lines of code
        snippet = md.get("symbol_path") or md.get("symbol") or ""
        code = (md.get("code") or "")[:500]
        txt = f"{snippet}\n{code}" if code else snippet
        if not txt:
            txt = (p.payload or {}).get("information") or ""
        pairs.append(f"{query} [SEP] {txt}")
    return pairs


def rerank_local(pairs: List[str]) -> List[float]:
    # Requires RERANKER_ONNX_PATH and RERANKER_TOKENIZER_PATH to be set
    if not (ort and Tokenizer and RERANKER_ONNX_PATH and RERANKER_TOKENIZER_PATH):
        return [0.0 for _ in pairs]
    tok = Tokenizer.from_file(RERANKER_TOKENIZER_PATH)
    enc = tok.encode_batch(pairs)
    input_ids = [e.ids for e in enc]
    attn = [e.attention_mask for e in enc]
    # Pad to max length
    max_len = max(len(ids) for ids in input_ids) if input_ids else 0
    def pad(seq, pad_id=0):
        return seq + [pad_id] * (max_len - len(seq))
    input_ids = [pad(s) for s in input_ids]
    attn = [pad(s) for s in attn]
    sess = ort.InferenceSession(RERANKER_ONNX_PATH, providers=["CPUExecutionProvider"])  # local CPU
    feeds = {
        sess.get_inputs()[0].name: input_ids,
        sess.get_inputs()[1].name: attn,
    }
    out = sess.run(None, feeds)
    # Heuristic: prefer the first output; accept scalar per row or 2-class logits
    logits = out[0]
    scores: List[float] = []
    for row in logits:
        if isinstance(row, list) and len(row) == 2:
            scores.append(float(row[1]))
        elif hasattr(row, "__len__") and len(row) == 1:
            scores.append(float(row[0]))
        else:
            # Fallback: mean of row
            try:
                scores.append(float(sum(row) / max(1, len(row))))
            except Exception:
                scores.append(0.0)
    return scores


def main():
    ap = argparse.ArgumentParser(description="Local cross-encoder rerank (ONNX)")
    ap.add_argument("--query", "-q", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--under", type=str, default=None)
    args = ap.parse_args()

    vec_name = _sanitize_vector_name(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    must = []
    if args.language:
        must.append(models.FieldCondition(key="metadata.language", match=models.MatchValue(value=args.language)))
    if args.under:
        must.append(models.FieldCondition(key="metadata.path_prefix", match=models.MatchValue(value=args.under)))
    flt = models.Filter(must=must) if must else None

    pts = dense_results(client, vec_name, args.query, flt, args.topk)
    if not pts:
        print("No results.")
        return
    pairs = prepare_pairs(args.query, pts)
    scores = rerank_local(pairs)
    # Combine original ordering with reranker scores (stable if all zeros)
    ranked = list(zip(scores, pts))
    ranked.sort(key=lambda x: x[0], reverse=True)
    for s, p in ranked[: args.limit]:
        md = (p.payload or {}).get("metadata") or {}
        print(f"{s:.3f}\t{md.get('path')}\t{md.get('symbol_path') or md.get('symbol') or ''}\t{md.get('start_line')}-{md.get('end_line')}")


if __name__ == "__main__":
    main()

