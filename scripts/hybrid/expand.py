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
    "CODE_SYNONYMS", "tokenize_queries",
    "expand_queries", "expand_queries_enhanced", "expand_queries_weighted",
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


# Phrase-level synonyms (applied first, before word-level)
# These handle common multi-word patterns that should be replaced as a unit
PHRASE_SYNONYMS: Dict[str, List[str]] = {
    "true or false": ["boolean", "bool"],
    "true and false": ["boolean", "bool"],
    "true/false": ["boolean", "bool"],
    "1 or 0": ["boolean", "binary"],
    "yes or no": ["boolean", "bool"],
    "key value": ["dict", "map", "dictionary"],
    "key-value": ["dict", "map", "dictionary"],
    "linked list": ["linkedlist", "list node"],
    "hash map": ["hashmap", "dict", "dictionary"],
    "hash table": ["hashtable", "dict", "dictionary"],
    "binary tree": ["binarytree", "tree node"],
    "try catch": ["exception handling", "error handling"],
    "try except": ["exception handling", "error handling"],
    "read file": ["file read", "open file", "load file"],
    "write file": ["file write", "save file"],
    "for loop": ["iterate", "loop", "iteration"],
    "while loop": ["loop", "iteration"],
    "if else": ["conditional", "branch"],
    "switch case": ["match", "conditional", "branch"],
}

# Cross-language code-aware query expansion synonyms
# Covers: Python, JavaScript/TypeScript, Java, C/C++, C#, Go, Rust, Ruby,
#         PHP, Swift, Kotlin, Scala, R, Shell/Bash, SQL, Lua
CODE_SYNONYMS: Dict[str, List[str]] = {
    # === Functions & Methods ===
    "function": ["method", "def", "fn", "func", "proc", "sub", "lambda"],
    "method": ["function", "fn", "member", "procedure"],
    "lambda": ["anonymous", "closure", "arrow", "callback"],
    "callback": ["handler", "listener", "hook", "delegate"],
    # === Classes & Types ===
    "class": ["type", "struct", "object", "interface", "trait"],
    "struct": ["class", "record", "type", "data"],
    "interface": ["protocol", "trait", "abstract", "contract"],
    "type": ["class", "struct", "typedef", "alias"],
    "object": ["instance", "entity", "class"],
    "enum": ["enumeration", "constant", "variant"],
    # === Creation & Initialization ===
    "create": ["init", "initialize", "construct", "new", "make", "build"],
    "new": ["create", "alloc", "instantiate", "spawn"],
    "init": ["initialize", "setup", "constructor", "create"],
    "build": ["create", "construct", "make", "generate"],
    # === Getters & Setters ===
    "get": ["fetch", "retrieve", "read", "load", "obtain", "query"],
    "set": ["assign", "update", "write", "store", "put"],
    "read": ["get", "load", "fetch", "input", "receive"],
    "write": ["set", "save", "store", "output", "send"],
    # === Type Conversion ===
    "cast": ["convert", "coerce", "transform", "parse", "as"],
    "convert": ["cast", "transform", "parse", "translate", "map"],
    "parse": ["convert", "extract", "decode", "deserialize", "read"],
    "format": ["stringify", "serialize", "encode", "render", "print"],
    "serialize": ["encode", "marshal", "dump", "stringify", "json"],
    "deserialize": ["decode", "unmarshal", "load", "parse"],
    # === Boolean & Logic ===
    "true": ["boolean", "bool", "yes", "truthy"],
    "false": ["boolean", "bool", "no", "falsy"],
    "boolean": ["bool", "logical", "flag", "predicate"],
    "check": ["validate", "verify", "test", "assert", "is"],
    "validate": ["check", "verify", "ensure", "assert"],
    # === Numeric ===
    "number": ["numeric", "int", "float", "integer", "digit"],
    "numbers": ["numeric", "int", "float", "integer", "digits"],
    "numeric": ["number", "int", "float", "decimal"],
    "integer": ["int", "number", "whole", "long"],
    "float": ["double", "decimal", "real", "number"],
    # === Strings & Text ===
    "string": ["str", "text", "char", "varchar"],
    "text": ["string", "content", "message", "data"],
    "concat": ["join", "append", "combine", "merge"],
    "split": ["separate", "divide", "tokenize", "explode"],
    "trim": ["strip", "clean", "remove", "whitespace"],
    "replace": ["substitute", "swap", "change", "update"],
    "match": ["regex", "pattern", "find", "search"],
    # === Collections ===
    "array": ["list", "vector", "slice", "sequence", "collection"],
    "list": ["array", "vector", "sequence", "collection"],
    "map": ["dict", "dictionary", "hash", "object", "table"],
    "dict": ["map", "dictionary", "hash", "object", "hashmap"],
    "set": ["hashset", "unique", "distinct", "collection"],
    "queue": ["fifo", "buffer", "deque", "channel"],
    "stack": ["lifo", "push", "pop"],
    # === Null & Empty ===
    "null": ["nil", "none", "undefined", "nothing", "void"],
    "empty": ["blank", "zero", "clear", "vacant", "nil"],
    "none": ["null", "nil", "nothing", "undefined"],
    "undefined": ["null", "unset", "missing", "void"],
    # === Error Handling ===
    "error": ["exception", "err", "failure", "fault"],
    "exception": ["error", "throw", "raise", "catch"],
    "throw": ["raise", "panic", "error", "emit"],
    "catch": ["except", "handle", "rescue", "recover"],
    "try": ["attempt", "catch", "handle"],
    # === Async & Concurrency ===
    "async": ["await", "promise", "future", "concurrent"],
    "await": ["async", "wait", "then", "resolve"],
    "promise": ["future", "async", "deferred", "task"],
    "thread": ["goroutine", "coroutine", "task", "worker"],
    "lock": ["mutex", "sync", "semaphore", "guard"],
    "channel": ["queue", "pipe", "stream", "buffer"],
    # === I/O & Files ===
    "file": ["path", "stream", "io", "fd"],
    "open": ["read", "load", "access", "fopen"],
    "close": ["dispose", "release", "cleanup", "shut"],
    "save": ["write", "store", "persist", "dump"],
    "load": ["read", "open", "fetch", "import"],
    # === HTTP & Network ===
    "request": ["req", "call", "fetch", "http"],
    "response": ["res", "reply", "result", "answer"],
    "send": ["post", "emit", "transmit", "write"],
    "receive": ["get", "read", "listen", "accept"],
    "connect": ["open", "establish", "join", "link"],
    # === Database ===
    "query": ["select", "find", "search", "fetch"],
    "insert": ["add", "create", "put", "store"],
    "update": ["modify", "change", "set", "patch"],
    "delete": ["remove", "drop", "destroy", "erase"],
    # === Control Flow ===
    "loop": ["iterate", "for", "while", "each", "repeat"],
    "iterate": ["loop", "traverse", "walk", "scan"],
    "filter": ["where", "select", "find", "grep"],
    "sort": ["order", "arrange", "rank", "compare"],
    "find": ["search", "lookup", "locate", "query", "get"],
    "search": ["find", "lookup", "query", "grep", "match"],
    # === Common Operations ===
    "add": ["append", "insert", "push", "put", "plus"],
    "remove": ["delete", "pop", "erase", "drop", "unset"],
    "copy": ["clone", "duplicate", "deep", "shallow"],
    "move": ["transfer", "relocate", "shift"],
    "merge": ["combine", "join", "concat", "union"],
    "compare": ["diff", "equal", "match", "cmp"],
    # === Testing ===
    "test": ["spec", "assert", "verify", "check", "unit"],
    "mock": ["stub", "fake", "spy", "double"],
    "assert": ["expect", "should", "verify", "check"],
}


def _expand_phrases(queries: List[str], max_extra: int = 2) -> List[str]:
    """Apply phrase-level synonym expansion."""
    phrase_expanded: List[str] = []
    for q in list(queries):
        ql = q.lower()
        for phrase, replacements in PHRASE_SYNONYMS.items():
            if phrase in ql:
                for repl in replacements[:max_extra]:
                    exp = ql.replace(phrase, repl)
                    if exp != ql and exp not in phrase_expanded:
                        phrase_expanded.append(exp)
    return phrase_expanded


def _expand_words(queries: List[str], max_extra: int = 2) -> List[str]:
    """Apply word-level synonym expansion."""
    word_expanded: List[str] = []
    for q in list(queries):
        ql = q.lower()
        for word, syns in CODE_SYNONYMS.items():
            if word in ql:
                for s in syns[:max_extra]:
                    exp = re.sub(rf"\b{re.escape(word)}\b", s, q, flags=re.IGNORECASE)
                    if exp not in queries and exp not in word_expanded:
                        word_expanded.append(exp)
    return word_expanded


def expand_queries(
    queries: List[str], language: str | None = None, max_extra: int = 2
) -> List[str]:
    """Expand queries using code-aware synonyms.

    Applies phrase-level synonyms first (e.g., "true or false" -> "boolean"),
    then word-level synonyms. Phrase expansions are prioritized at the front
    of the list since they often produce better semantic matches.

    Args:
        queries: Original query strings
        language: Optional programming language hint (currently unused)
        max_extra: Maximum number of synonym expansions per word

    Returns:
        List of expanded queries with phrase expansions first, then originals,
        then word-level expansions
    """
    phrase_expanded = _expand_phrases(queries, max_extra)
    word_expanded = _expand_words(queries, max_extra)
    # Filter word expansions that overlap with phrase expansions
    word_expanded = [w for w in word_expanded if w not in phrase_expanded]

    # Order: phrase expansions first (best semantic match), then originals, then word expansions
    out: List[str] = phrase_expanded + list(queries) + word_expanded

    # Dedupe while preserving order
    seen: set = set()
    deduped: List[str] = []
    for q in out:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    # Allow more expansions - fusion weighting will handle prioritization
    max_queries = max(12, len(queries) * 4)
    return deduped[:max_queries]


def expand_queries_weighted(
    queries: List[str],
    language: str | None = None,
    max_extra: int = 2,
    client: "QdrantClient | None" = None,
    model: Any = None,
    collection: str | None = None,
) -> tuple[List[str], List[float]]:
    """
    Enhanced query expansion returning queries with fusion weights.

    Weights reflect query quality tiers:
    - Tier 1 (1.0): Phrase-expanded queries (best semantic match)
    - Tier 2 (0.85): Original queries
    - Tier 3 (0.6): Word-substituted queries
    - Tier 4 (0.4): Semantic expansion queries

    Returns:
        Tuple of (queries, weights) where weights indicate fusion priority
    """
    original_set = set(queries)
    out: List[str] = []
    weights: List[float] = []
    seen: set = set()

    # 1. Get phrase expansions (Tier 1 - weight 1.0)
    phrase_expanded = _expand_phrases(queries)
    for q in phrase_expanded:
        if q not in seen and q not in original_set:
            seen.add(q)
            out.append(q)
            weights.append(1.0)

    # 2. Original queries - weight depends on whether phrase expansion found matches
    # If phrase expansion produced results, originals are Tier 2 (0.85)
    # If no phrase expansion, originals are Tier 1 (1.0) - they're the best we have
    original_weight = 0.85 if phrase_expanded else 1.0
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
            weights.append(original_weight)

    # 3. Get word expansions (Tier 3 - weight 0.6)
    word_expanded = _expand_words(queries, max_extra)
    for q in word_expanded:
        if q not in seen:
            seen.add(q)
            out.append(q)
            weights.append(0.6)

    # 4. Semantic expansion (Tier 4 - weight 0.4)
    if SEMANTIC_EXPANSION_AVAILABLE and expand_queries_semantically and client and model:
        try:
            semantic_terms = expand_queries_semantically(
                queries, language, client, model, collection, max_extra
            )
            for q in list(queries):
                for term in semantic_terms:
                    if term not in seen:
                        seen.add(term)
                        out.append(term)
                        weights.append(0.4)
                    combined = f"{q} {term}"
                    if combined not in seen:
                        seen.add(combined)
                        out.append(combined)
                        weights.append(0.4)
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Semantic expansion added {len(semantic_terms)} terms: {semantic_terms}")
        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                logger.debug(f"Semantic expansion failed: {e}")

    # Limit total queries
    max_queries = max(12, len(queries) * 4)
    return out[:max_queries], weights[:max_queries]


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

    Wrapper around expand_queries_weighted that returns just the queries.
    """
    expanded, _ = expand_queries_weighted(queries, language, max_extra, client, model, collection)
    return expanded


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

