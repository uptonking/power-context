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
    # Fallback: compact name
    return name.replace("/", "-").replace("_", "-")[:64]


# Shared lexical hashing utilities to keep ingest/search/memory consistent
import re, hashlib, math


def _split_ident_lex(s: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x]


def lex_hash_vector_text(text: str, dim: int = 4096) -> list[float]:
    if not text:
        return [0.0] * dim
    toks = _split_ident_lex(text)
    if not toks:
        return [0.0] * dim
    vec = [0.0] * dim
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def lex_hash_vector_queries(phrases: list[str], dim: int = 4096) -> list[float]:
    toks: list[str] = []
    for ph in phrases or []:
        toks.extend(_split_ident_lex(str(ph)))
    if not toks:
        return [0.0] * dim
    vec = [0.0] * dim
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
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
