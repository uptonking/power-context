#!/usr/bin/env python3
from __future__ import annotations

# Canonical vector-name sanitizer used by both ingest and search paths
# Keeps mapping consistent across the codebase


def sanitize_vector_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    # Common fastembed alias mapping for MiniLM
    if name in (
        "sentence-transformers/all-minilm-l6-v2",
        "sentence-transformers/all-minilm-l-6-v2",
        "sentence-transformers/all-minilm-l6-v2",
    ):
        return "fast-all-minilm-l6-v2"
    # Common fastembed alias mapping for BGE base
    if "bge-base-en-v1.5" in name:
        return "fast-bge-base-en-v1.5"
    # Qwen3-Embedding ONNX model
    if "qwen3-embedding" in name:
        return "fast-qwen3-embedding-0.6b"
    # Fallback: compact name
    return name.replace("/", "-").replace("_", "-")[:64]


# Shared lexical hashing utilities to keep ingest/search/memory consistent
import re, hashlib, math, os

# Feature flags for improved lexical hashing (v2)
# LEX_MULTI_HASH: number of hash functions per token (default 3 for v2, 1 for legacy)
# LEX_BIGRAMS: enable bigram hashing (default 1 for v2)
_LEX_MULTI_HASH = int(os.environ.get("LEX_MULTI_HASH", "3") or 3)
_LEX_BIGRAMS = os.environ.get("LEX_BIGRAMS", "1").strip().lower() in ("1", "true", "yes", "on")
_LEX_BIGRAM_WEIGHT = float(os.environ.get("LEX_BIGRAM_WEIGHT", "0.7") or 0.7)


def _split_ident_lex(s: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x]


def _hash_token(token: str, seed: int = 0) -> int:
    """Hash a token with optional seed for multi-hash."""
    if seed == 0:
        return int(hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
    return int(hashlib.md5(f"{seed}:{token}".encode("utf-8", errors="ignore")).hexdigest()[:8], 16)


def lex_hash_vector_text(text: str, dim: int = 2048) -> list[float]:
    """Hash text into sparse lexical vector with multi-hash and bigrams.

    Improvements over v1:
    - Multi-hash: each token hashes to multiple buckets (reduces collision impact)
    - Bigrams: consecutive token pairs captured for phrase matching
    """
    if not text:
        return [0.0] * dim
    toks = _split_ident_lex(text)
    if not toks:
        return [0.0] * dim

    vec = [0.0] * dim
    n_hashes = _LEX_MULTI_HASH

    # Unigrams with multi-hash
    for t in toks:
        for seed in range(n_hashes):
            h = _hash_token(t, seed)
            vec[h % dim] += 1.0

    # Bigrams (weighted less than unigrams)
    if _LEX_BIGRAMS and len(toks) > 1:
        for i in range(len(toks) - 1):
            bigram = f"{toks[i]}_{toks[i+1]}"
            for seed in range(n_hashes):
                h = _hash_token(bigram, seed)
                vec[h % dim] += _LEX_BIGRAM_WEIGHT

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def lex_hash_vector_queries(phrases: list[str], dim: int = 2048) -> list[float]:
    """Hash query phrases into sparse lexical vector (same algorithm as text)."""
    toks: list[str] = []
    for ph in phrases or []:
        toks.extend(_split_ident_lex(str(ph)))
    if not toks:
        return [0.0] * dim

    vec = [0.0] * dim
    n_hashes = _LEX_MULTI_HASH

    # Unigrams with multi-hash
    for t in toks:
        for seed in range(n_hashes):
            h = _hash_token(t, seed)
            vec[h % dim] += 1.0

    # Bigrams
    if _LEX_BIGRAMS and len(toks) > 1:
        for i in range(len(toks) - 1):
            bigram = f"{toks[i]}_{toks[i+1]}"
            for seed in range(n_hashes):
                h = _hash_token(bigram, seed)
                vec[h % dim] += _LEX_BIGRAM_WEIGHT

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# Shared snippet highlighter used by servers and CLIs
# Highlights tokens by wrapping matches with << >> (longest-first, case-insensitive)
def highlight_snippet(snippet: str, tokens: list[str]) -> str:
    if not snippet or not tokens:
        return snippet

    # longest first to avoid partial overlaps
    toks = sorted(set(tokens), key=len, reverse=True)
    import re as _re

    def _repl(m):
        return f"<<{m.group(0)}>>"

    for t in toks:
        try:
            pat = _re.compile(_re.escape(t), _re.IGNORECASE)
            snippet = pat.sub(_repl, snippet)
        except Exception:
            continue
    return snippet
