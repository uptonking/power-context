#!/usr/bin/env python3
"""
Query expansion logic for hybrid search.

This module provides query expansion functionality including:
- Synonym-based expansion (CODE_SYNONYMS)
- Basic expansion (expand_queries)
- Enhanced semantic expansion (expand_queries_enhanced)
- LLM-assisted expansion (_llm_expand_queries)
- Pseudo-relevance feedback (PRF) terms extraction (_prf_terms_from_results)
"""

__all__ = [
    "CODE_SYNONYMS", "expand_queries", "expand_queries_enhanced",
    "_llm_expand_queries", "_prf_terms_from_results",
    "SEMANTIC_EXPANSION_AVAILABLE",
    "expand_queries_semantically", "expand_queries_with_prf",
    "get_expansion_stats", "clear_expansion_cache",
]

import os
import re
import ast
import json
import logging
from typing import List, Dict, Any, TYPE_CHECKING
from pathlib import Path

logger = logging.getLogger("hybrid_expand")

# Import QdrantClient type for annotations
if TYPE_CHECKING:
    from qdrant_client import QdrantClient

# Import semantic expansion functionality (optional)
try:
    from scripts.semantic_expansion import (
        expand_queries_semantically,
        expand_queries_with_prf,
        get_expansion_stats,
        clear_expansion_cache,
    )
    SEMANTIC_EXPANSION_AVAILABLE = True
except ImportError:
    SEMANTIC_EXPANSION_AVAILABLE = False
    expand_queries_semantically = None
    expand_queries_with_prf = None
    get_expansion_stats = None
    clear_expansion_cache = None


# Stop words for tokenization
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "to",
    "with",
    "by",
    "is",
    "are",
    "be",
    "this",
    "that",
}


def _split_ident(s: str) -> List[str]:
    """Split snake_case and camelCase identifiers into tokens."""
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        # camelCase split
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def tokenize_queries(phrases: List[str]) -> List[str]:
    """Tokenize query phrases into individual tokens.
    
    Splits identifiers on camelCase and snake_case boundaries,
    removes stop words, and deduplicates while preserving order.
    """
    toks: List[str] = []
    for ph in phrases:
        toks.extend(_split_ident(ph))
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# Minimal code-aware query expansion synonyms
CODE_SYNONYMS: Dict[str, List[str]] = {
    "function": ["method", "def", "fn"],
    "class": ["type", "object"],
    "create": ["init", "initialize", "construct"],
    "get": ["fetch", "retrieve"],
    "set": ["assign", "update"],
}


def expand_queries(
    queries: List[str], language: str | None = None, max_extra: int = 2
) -> List[str]:
    """Expand queries using code-aware synonyms.
    
    Args:
        queries: Original query strings
        language: Optional programming language hint (currently unused)
        max_extra: Maximum number of synonym expansions per word
        
    Returns:
        List of expanded queries including originals and synonym variants
    """
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


def expand_queries_enhanced(
    queries: List[str],
    language: str | None = None,
    max_extra: int = 2,
    client: "QdrantClient | None" = None,
    model: Any = None,
    collection: str | None = None,
) -> List[str]:
    """
    Enhanced query expansion combining synonym-based and semantic similarity approaches.

    Args:
        queries: Original query strings
        language: Optional programming language hint
        max_extra: Maximum number of additional expansions per query
        client: QdrantClient instance for semantic expansion
        model: Embedding model instance for semantic analysis
        collection: Collection name for semantic expansion

    Returns:
        List of expanded queries
    """
    # Start with original queries
    out: List[str] = list(queries)

    # 1. Apply traditional synonym-based expansion
    synonym_expanded = expand_queries(queries, language, max_extra)
    for q in synonym_expanded:
        if q not in out:
            out.append(q)

    # 2. Apply semantic similarity expansion if available
    if SEMANTIC_EXPANSION_AVAILABLE and expand_queries_semantically and client and model:
        try:
            semantic_terms = expand_queries_semantically(
                queries, language, client, model, collection, max_extra
            )

            # Create expanded queries using semantic terms
            for q in list(queries):
                for term in semantic_terms:
                    # Add term as a standalone query
                    if term not in out:
                        out.append(term)

                    # Create combined queries with semantic terms
                    combined = f"{q} {term}"
                    if combined not in out:
                        out.append(combined)

            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Semantic expansion added {len(semantic_terms)} terms: {semantic_terms}")

        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Semantic expansion failed: {e}")

    # Limit total number of queries to prevent explosion
    max_queries = max(8, len(queries) * 3)
    return out[:max_queries]


def _llm_expand_queries(
    queries: List[str], language: str | None = None, max_new: int = 4
) -> List[str]:
    """Best-effort LLM expansion using configured decoder.
    
    If REFRAG_RUNTIME is set, uses the configured client (glm, minimax, llamacpp).
    If REFRAG_RUNTIME is unset, tries llamacpp (for users with just the container).
    On any error, returns [] silently.
    
    Args:
        queries: Original query strings
        language: Optional programming language hint (currently unused)
        max_new: Maximum number of new query alternatives to generate
        
    Returns:
        List of alternative query phrasings, empty on error
    """
    if not queries or max_new <= 0:
        return []

    # If REFRAG_RUNTIME is explicitly set, use it; otherwise default to llamacpp
    runtime_kind = os.environ.get("REFRAG_RUNTIME", "").strip().lower() or "llamacpp"
    
    original_q = " ".join(queries)
    
    def _parse_alts(out: str) -> List[str]:
        """Parse alternatives from LLM output."""
        alts: List[str] = []
        # Try direct JSON parse (strip markdown code fences if present)
        try:
            from scripts.llm_utils import strip_markdown_fences
            parsed = json.loads(strip_markdown_fences(out))
            if isinstance(parsed, list):
                for s in parsed:
                    if isinstance(s, str) and s.strip() and s not in queries:
                        alts.append(s.strip())
                        if len(alts) >= max_new:
                            return alts
        except Exception:
            pass
        # Try ast.literal_eval for single-quoted lists
        try:
            parsed = ast.literal_eval(out)
            if isinstance(parsed, list):
                for s in parsed:
                    if isinstance(s, str) and s.strip() and s not in queries:
                        alts.append(s.strip())
                        if len(alts) >= max_new:
                            return alts
        except Exception:
            pass
        # Try regex extraction from verbose output - only keep multi-word phrases
        for m in re.finditer(r'"([^"]+)"', out):
            candidate = m.group(1).strip()
            # Skip single words and duplicates - we want complete search phrases
            if candidate and " " in candidate and candidate not in queries and candidate not in alts:
                alts.append(candidate)
                if len(alts) >= max_new:
                    break
        return alts

    try:
        max_tokens = int(os.environ.get("EXPAND_MAX_TOKENS", "512"))
        if runtime_kind == "glm":
            from scripts.refrag_glm import GLMRefragClient
            client = GLMRefragClient()
            prompt = f'Rewrite "{original_q}" as {max_new} different code search queries using synonyms or related terms. Each query should be a complete phrase, not single words. Output as JSON array:'
            txt = client.generate_with_soft_embeddings(
                prompt, max_tokens=max_tokens, temperature=1.0, top_p=0.9,
                disable_thinking=True, force_json=False
            )
        elif runtime_kind == "minimax":
            from scripts.refrag_minimax import MiniMaxRefragClient
            client = MiniMaxRefragClient()
            prompt = f'Rewrite "{original_q}" as {max_new} different search queries using synonyms:'
            txt = client.generate_with_soft_embeddings(
                prompt, max_tokens=max_tokens, temperature=1.0,
                system="You rewrite search queries using synonyms. Output format: JSON array of strings. No other text."
            )
        else:
            from scripts.refrag_llamacpp import LlamaCppRefragClient
            client = LlamaCppRefragClient()
            prompt = (
                f"Rewrite this code search query using different words: {original_q}\n"
                f'Give {max_new} short alternative phrasings as a JSON array. Example: ["alt1", "alt2"]'
            )
            txt = client.generate_with_soft_embeddings(prompt, max_tokens=max_tokens, temperature=0.7)
        return _parse_alts(txt)
    except Exception:
        return []


def _prf_terms_from_results(
    score_map: Dict[str, Dict[str, Any]], top_docs: int = 8, max_terms: int = 6
) -> List[str]:
    """Extract pseudo-relevant feedback terms from top documents' metadata.
    
    Uses the tokenize_queries function to extract tokens from symbol names
    and paths of the top-scoring documents.
    
    Args:
        score_map: Dictionary mapping point IDs to scoring records with 'pt' and 's' keys
        top_docs: Number of top documents to extract terms from
        max_terms: Maximum number of terms to return
        
    Returns:
        List of most frequent tokens from top documents
    """
    # Rank by current fused score 's'
    ranked = sorted(score_map.values(), key=lambda r: r.get("s", 0.0), reverse=True)[
        : max(1, top_docs)
    ]
    freq: Dict[str, int] = {}
    for rec in ranked:
        pt = rec.get("pt")
        if pt is None:
            continue
        payload = getattr(pt, "payload", None) or {}
        md = payload.get("metadata") or {}
        path = str(md.get("path") or md.get("symbol_path") or md.get("file_path") or "")
        symbol = str(md.get("symbol") or "")
        text = f"{symbol} {path}"
        for tok in tokenize_queries([text]):
            if tok:
                freq[tok] = freq.get(tok, 0) + 1
    # sort by frequency desc
    terms = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in terms[: max(1, max_terms)]]


# Re-export semantic expansion functions if available
__all__ = [
    "CODE_SYNONYMS",
    "tokenize_queries",
    "expand_queries",
    "expand_queries_enhanced",
    "_llm_expand_queries",
    "_prf_terms_from_results",
    "SEMANTIC_EXPANSION_AVAILABLE",
]

# Conditionally export semantic expansion functions
if SEMANTIC_EXPANSION_AVAILABLE:
    __all__.extend([
        "expand_queries_semantically",
        "expand_queries_with_prf", 
        "get_expansion_stats",
        "clear_expansion_cache",
    ])
