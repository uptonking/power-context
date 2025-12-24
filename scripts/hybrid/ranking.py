#!/usr/bin/env python3
"""
Ranking and scoring logic for hybrid search.

This module extracts the core ranking, scoring, and diversification functions
from hybrid_search.py for reuse and testing.
"""

__all__ = [
    "rrf", "_scale_rrf_k", "_adaptive_per_query", "_normalize_scores",
    "sparse_lex_score", "lexical_score",
    "_compute_query_stats", "_adaptive_weights", "_bm25_token_weights_from_results",
    "_mmr_diversify", "_merge_and_budget_spans",
    "_detect_implementation_intent", "_IMPL_INTENT_PATTERNS",
    "_get_collection_stats", "_COLL_STATS_CACHE", "_COLL_STATS_TTL",
]

import os
import re
import math
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("hybrid_ranking")

# ---------------------------------------------------------------------------
# Helper: safe type coercion
# ---------------------------------------------------------------------------

def _safe_int(val: Any, default: int) -> int:
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float) -> float:
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Configuration constants (loaded from environment with defaults)
# ---------------------------------------------------------------------------

RRF_K = _safe_int(os.environ.get("HYBRID_RRF_K", "30"), 30)
DENSE_WEIGHT = _safe_float(os.environ.get("HYBRID_DENSE_WEIGHT", "1.5"), 1.5)
LEXICAL_WEIGHT = _safe_float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "0.20"), 0.20)
LEX_VECTOR_WEIGHT = _safe_float(
    os.environ.get("HYBRID_LEX_VECTOR_WEIGHT", str(LEXICAL_WEIGHT)), LEXICAL_WEIGHT
)
PSEUDO_BOOST = _safe_float(os.environ.get("HYBRID_PSEUDO_BOOST", "0.0"), 0.0)

# Large codebase scaling
LARGE_COLLECTION_THRESHOLD = _safe_int(os.environ.get("HYBRID_LARGE_THRESHOLD", "10000"), 10000)
MAX_RRF_K_SCALE = _safe_float(os.environ.get("HYBRID_MAX_RRF_K_SCALE", "3.0"), 3.0)
SCORE_NORMALIZE_ENABLED = os.environ.get("HYBRID_SCORE_NORMALIZE", "1").lower() in {"1", "true", "yes", "on"}

# Sparse lexical vector scoring
SPARSE_LEX_MAX_SCORE = _safe_float(os.environ.get("HYBRID_SPARSE_LEX_MAX", "15.0"), 15.0)
SPARSE_RRF_MAX = 1.0 / (RRF_K + 1)
SPARSE_RRF_MIN = 1.0 / (RRF_K + 50)

# Micro-span budgeting defaults
def _get_micro_defaults() -> Tuple[int, int, int, int]:
    """Return (max_spans, merge_lines, budget_tokens, tokens_per_line) based on runtime and micro chunk mode.

    Budget tokens floor is 5000 to ensure context_answer has enough context for quality answers.
    """
    micro_enabled = os.environ.get("INDEX_MICRO_CHUNKS", "1").strip().lower() in {"1", "true", "yes", "on"}
    try:
        from scripts.refrag_glm import detect_glm_runtime
        is_glm = detect_glm_runtime()
    except ImportError:
        is_glm = False
    if is_glm:
        if micro_enabled:
            return (24, 6, 8192, 32)
        else:
            return (12, 4, 6000, 32)
    else:
        # Non-GLM: still need reasonable budget for quality context_answer
        return (8, 4, 5000, 32)

_MICRO_DEFAULTS = _get_micro_defaults()
MICRO_OUT_MAX_SPANS = _safe_int(os.environ.get("MICRO_OUT_MAX_SPANS", str(_MICRO_DEFAULTS[0])), _MICRO_DEFAULTS[0])
MICRO_MERGE_LINES = _safe_int(os.environ.get("MICRO_MERGE_LINES", str(_MICRO_DEFAULTS[1])), _MICRO_DEFAULTS[1])
MICRO_BUDGET_TOKENS = _safe_int(os.environ.get("MICRO_BUDGET_TOKENS", str(_MICRO_DEFAULTS[2])), _MICRO_DEFAULTS[2])
MICRO_TOKENS_PER_LINE = _safe_int(os.environ.get("MICRO_TOKENS_PER_LINE", str(_MICRO_DEFAULTS[3])), _MICRO_DEFAULTS[3])

# Intent detection for implementation preference
INTENT_IMPL_BOOST = _safe_float(os.environ.get("HYBRID_INTENT_IMPL_BOOST", "0.15"), 0.15)

_IMPL_INTENT_PATTERNS = frozenset({
    "implementation", "how does", "how is", "where is", "code for",
    "function that", "method that", "class that", "implements",
    "defined", "definition", "source", "logic", "algorithm",
    "where", "find", "locate", "show me", "actual code",
})

# ---------------------------------------------------------------------------
# Collection stats cache (for large collection scaling)
# ---------------------------------------------------------------------------

_COLL_STATS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_COLL_STATS_TTL = 300  # 5 minutes


def _get_collection_stats(client: Any, coll_name: str) -> Dict[str, Any]:
    """Get cached collection statistics for scaling decisions."""
    import time
    now = time.time()
    cached = _COLL_STATS_CACHE.get(coll_name)
    if cached and (now - cached[0]) < _COLL_STATS_TTL:
        return cached[1]
    try:
        info = client.get_collection(coll_name)
        stats = {"points_count": info.points_count or 0}
        _COLL_STATS_CACHE[coll_name] = (now, stats)
        return stats
    except Exception:
        return {"points_count": 0}


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def rrf(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a given rank."""
    return 1.0 / (k + rank)


def _scale_rrf_k(base_k: int, collection_size: int) -> int:
    """Scale RRF k parameter based on collection size.

    For large collections, increase k to spread score distribution.
    Uses logarithmic scaling: k_scaled = k * (1 + log10(size/threshold))
    Capped at MAX_RRF_K_SCALE * base_k.
    """
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return base_k
    ratio = collection_size / LARGE_COLLECTION_THRESHOLD
    scale = 1.0 + math.log10(max(1, ratio))
    scale = min(scale, MAX_RRF_K_SCALE)
    return int(base_k * scale)


def _adaptive_per_query(base_limit: int, collection_size: int, has_filters: bool) -> int:
    """Increase candidate retrieval for larger collections.

    Uses sublinear sqrt scaling to avoid excessive retrieval.
    Filters reduce the need for extra candidates.
    """
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return base_limit
    ratio = collection_size / LARGE_COLLECTION_THRESHOLD
    scale = math.sqrt(ratio)
    if has_filters:
        scale = max(1.0, scale * 0.7)
    scaled = int(base_limit * min(scale, 3.0))
    return max(base_limit, min(scaled, 200))


def _normalize_scores(score_map: Dict[str, Dict[str, Any]], collection_size: int) -> None:
    """Normalize scores using z-score + sigmoid for large collections.

    This spreads compressed score distributions to improve discrimination.
    Only applies when SCORE_NORMALIZE_ENABLED=true and collection is large.
    """
    if not SCORE_NORMALIZE_ENABLED:
        return
    if collection_size < LARGE_COLLECTION_THRESHOLD:
        return
    if len(score_map) < 3:
        return

    scores = [rec["s"] for rec in score_map.values()]
    mean_s = sum(scores) / len(scores)
    var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    std_s = math.sqrt(var_s) if var_s > 0 else 1.0

    if std_s < 1e-6:
        return

    for rec in score_map.values():
        z = (rec["s"] - mean_s) / std_s
        normalized = 1.0 / (1.0 + math.exp(-z * 0.5))
        rec["s"] = normalized


# ---------------------------------------------------------------------------
# Sparse lexical scoring
# ---------------------------------------------------------------------------

def sparse_lex_score(raw_score: float, weight: float = LEX_VECTOR_WEIGHT) -> float:
    """Normalize sparse lexical vector score to RRF-equivalent range.

    Maps sparse similarity scores to the same range as RRF(rank) scores,
    preserving relative ordering while maintaining fusion balance.

    Formula: weight * (RRF_MIN + (clamped_score / max_score) * (RRF_MAX - RRF_MIN))
    - Sparse score 0 maps to RRF_MIN (like worst rank)
    - Sparse score max maps to RRF_MAX (like rank 1)
    - Quality ordering preserved, but doesn't dominate dense embedding scores
    """
    if raw_score <= 0:
        return 0.0
    clamped = min(raw_score, SPARSE_LEX_MAX_SCORE)
    ratio = clamped / SPARSE_LEX_MAX_SCORE
    rrf_equiv = SPARSE_RRF_MIN + ratio * (SPARSE_RRF_MAX - SPARSE_RRF_MIN)
    return weight * rrf_equiv


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to",
    "with", "by", "is", "are", "be", "this", "that",
}


def _split_ident(s: str) -> List[str]:
    """Split snake_case and camelCase identifiers into tokens."""
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def tokenize_queries(phrases: List[str]) -> List[str]:
    """Tokenize phrases into unique identifier-aware tokens."""
    toks: List[str] = []
    for ph in phrases:
        toks.extend(_split_ident(ph))
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# ---------------------------------------------------------------------------
# Lexical scoring
# ---------------------------------------------------------------------------

def lexical_score(
    phrases: List[str],
    md: Dict[str, Any],
    token_weights: Dict[str, float] | None = None,
    bm25_weight: float | None = None
) -> float:
    """Smarter lexical: split identifiers, weight matches in symbol/path higher.
    
    If token_weights provided, apply a small BM25-style multiplicative factor per token:
        factor = 1 + bm25_weight * (w - 1) where w are normalized around 1.0
    """
    tokens = tokenize_queries(phrases)
    if not tokens:
        return 0.0
    path = str(md.get("path", "")).lower()
    path_segs = re.split(r"[/\\]", path)
    sym = str(md.get("symbol", "")).lower()
    symp = str(md.get("symbol_path", "")).lower()
    code = str(md.get("code", ""))[:2000].lower()
    pseudo = str(md.get("pseudo") or "").lower()
    tags_val = md.get("tags") or []
    if isinstance(tags_val, list):
        tags_text = " ".join(str(x) for x in tags_val).lower()
    else:
        tags_text = str(tags_val).lower()
    s = 0.0
    for t in tokens:
        if not t:
            continue
        contrib = 0.0
        if t in sym or t in symp:
            contrib += 2.0
        if any(t in seg for seg in path_segs):
            contrib += 0.8
            if path_segs and t in path_segs[-1]:
                contrib += 0.3
        if t in code:
            contrib += 1.0
        if PSEUDO_BOOST > 0.0:
            if pseudo and t in pseudo:
                contrib += PSEUDO_BOOST
            if tags_text and t in tags_text:
                contrib += 0.5 * PSEUDO_BOOST
        if contrib > 0 and token_weights and bm25_weight:
            w = float(token_weights.get(t, 1.0) or 1.0)
            contrib *= (1.0 + float(bm25_weight) * (w - 1.0))
        s += contrib
    return s


# ---------------------------------------------------------------------------
# Adaptive weights
# ---------------------------------------------------------------------------

def _compute_query_stats(queries: List[str]) -> Dict[str, Any]:
    """Compute statistics about query tokens for adaptive weighting."""
    toks = tokenize_queries(queries)
    total = len(toks)

    def _is_camel(t: str) -> bool:
        try:
            return any(c.isupper() for c in t[1:]) and any(c.islower() for c in t)
        except Exception:
            return False

    def _is_identifier_like(t: str) -> bool:
        try:
            return ("_" in t) or t.isupper() or any(ch.isdigit() for ch in t) or _is_camel(t)
        except Exception:
            return False

    id_like = sum(1 for t in toks if _is_identifier_like(t))
    avg_tok_len = (sum(len(t) for t in toks) / max(1, total)) if total else 0.0
    qchars = sum(len(q) for q in queries) if queries else 0
    has_question = any(("?" in q) for q in (queries or []))
    q0 = (queries[0].strip().lower() if queries else "")
    wh_start = q0.startswith(("how", "what", "why", "when", "where", "explain", "describe"))
    stats = {
        "total_tokens": total,
        "identifier_density": (id_like / max(1, total)),
        "avg_token_len": avg_tok_len,
        "avg_query_chars": (qchars / max(1, len(queries))) if queries else 0.0,
        "narrative_hint": bool(has_question or wh_start),
    }
    return stats


def _adaptive_weights(stats: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return per-query weights (dense_w, lex_vec_w, lex_text_w) with gentle clamps.
    
    Dense/lex-vector vary within ±25%; lexical text component within ±20%.
    """
    base_d = DENSE_WEIGHT
    base_lv = LEX_VECTOR_WEIGHT
    base_lx = LEXICAL_WEIGHT

    id_density = float(stats.get("identifier_density", 0.0) or 0.0)
    total = int(stats.get("total_tokens", 0) or 0)
    narrative_hint = 1.0 if stats.get("narrative_hint") else 0.0
    longish = 1.0 if total >= 8 else 0.0

    narrative_score = 0.6 * narrative_hint + 0.4 * longish
    id_score = id_density
    delta = max(-1.0, min(1.0, narrative_score - id_score))

    dens_scale = 1.0 + 0.25 * delta
    lv_scale = 1.0 - 0.25 * delta
    lx_scale = 1.0 + 0.20 * (-delta)

    dens_scale = max(0.75, min(1.25, dens_scale))
    lv_scale = max(0.75, min(1.25, lv_scale))
    lx_scale = max(0.80, min(1.20, lx_scale))

    return base_d * dens_scale, base_lv * lv_scale, base_lx * lx_scale


def _bm25_token_weights_from_results(phrases: List[str], results: List[Any]) -> Dict[str, float]:
    """Compute lightweight per-token IDF-like weights from a small sample of lex results.
    
    Returns weights normalized to mean 1.0 over tokens present in phrases.
    """
    try:
        tokens = [t for t in tokenize_queries(phrases) if t]
        if not tokens or not results:
            return {}
        tok_set = set(tokens)
        N = max(1, len(results))
        df: Dict[str, int] = {t: 0 for t in tok_set}
        for p in results:
            try:
                md = (p.payload or {}).get("metadata") or {}
            except Exception:
                md = {}
            text = " ".join([
                str(md.get("symbol") or ""),
                str(md.get("symbol_path") or ""),
                str(md.get("path") or ""),
                str((md.get("code") or ""))[:2000],
            ]).lower()
            doc_toks = set(tokenize_queries([text]))
            for t in tok_set:
                if t in doc_toks:
                    df[t] += 1
        idf: Dict[str, float] = {t: math.log(1.0 + (N / float(df[t] + 1))) for t in tok_set}
        mean = sum(idf.values()) / max(1, len(idf))
        if mean <= 0:
            return {t: 1.0 for t in tok_set}
        return {t: (idf[t] / mean) for t in tok_set}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MMR diversification
# ---------------------------------------------------------------------------

def _mmr_diversify(ranked: List[Dict[str, Any]], k: int = 60, lambda_: float = 0.7) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance over fused list.
    
    Preserves top-1 by relevance, then balances relevance vs. diversity by path/symbol.
    Returns a reordered list (top-k diversified, remainder appended in original order).
    """
    if not ranked:
        return []
    k = max(1, min(int(k or 1), len(ranked)))

    def _path(md: Dict[str, Any]) -> str:
        return str(md.get("path") or "")

    def _symp(md: Dict[str, Any]) -> str:
        return str(md.get("symbol_path") or md.get("symbol") or "")

    def _sim(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        mda = (a["pt"].payload or {}).get("metadata") or {}
        mdb = (b["pt"].payload or {}).get("metadata") or {}
        pa, pb = _path(mda), _path(mdb)
        if pa and pb and pa == pb:
            return 1.0
        sa, sb = _symp(mda), _symp(mdb)
        if sa and sb and sa == sb:
            return 0.8
        if pa and pb:
            ta = set(re.split(r"[/\\]+", pa.lower()))
            tb = set(re.split(r"[/\\]+", pb.lower()))
            ta.discard("")
            tb.discard("")
            if ta and tb:
                inter = len(ta & tb)
                union = max(1, len(ta | tb))
                return 0.5 * (inter / union)
        return 0.0

    rel = [float(m.get("s", 0.0)) for m in ranked]
    selected_idx = [0]
    candidates = list(range(1, len(ranked)))
    while len(selected_idx) < k and candidates:
        best_idx = None
        best_score = -1e18
        for i in candidates:
            r = rel[i]
            if selected_idx:
                max_sim = max(_sim(ranked[i], ranked[j]) for j in selected_idx)
            else:
                max_sim = 0.0
            mmr = lambda_ * r - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected_idx.append(best_idx)
        candidates.remove(best_idx)

    sel_set = set(selected_idx)
    diversified = [ranked[i] for i in selected_idx]
    diversified.extend([ranked[i] for i in range(len(ranked)) if i not in sel_set])
    return diversified


# ---------------------------------------------------------------------------
# Micro-span budgeting
# ---------------------------------------------------------------------------

def _merge_and_budget_spans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Given ranked items with metadata path/start_line/end_line, merge nearby spans
    per path and enforce a token budget using a simple tokens-per-line estimate.
    
    Returns a filtered/merged list preserving score order as much as possible.
    """
    try:
        merge_lines = int(os.environ.get("MICRO_MERGE_LINES", str(MICRO_MERGE_LINES)) or MICRO_MERGE_LINES)
    except (ValueError, TypeError):
        merge_lines = MICRO_MERGE_LINES
    try:
        budget_tokens = int(os.environ.get("MICRO_BUDGET_TOKENS", str(MICRO_BUDGET_TOKENS)) or MICRO_BUDGET_TOKENS)
    except (ValueError, TypeError):
        budget_tokens = MICRO_BUDGET_TOKENS
    try:
        tokens_per_line = int(os.environ.get("MICRO_TOKENS_PER_LINE", str(MICRO_TOKENS_PER_LINE)) or MICRO_TOKENS_PER_LINE)
    except (ValueError, TypeError):
        tokens_per_line = MICRO_TOKENS_PER_LINE
    try:
        out_max_spans = int(os.environ.get("MICRO_OUT_MAX_SPANS", str(MICRO_OUT_MAX_SPANS)) or MICRO_OUT_MAX_SPANS)
    except (ValueError, TypeError):
        out_max_spans = MICRO_OUT_MAX_SPANS

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for m in items:
        md = {}
        try:
            if isinstance(m, dict):
                if m.get("path") or m.get("start_line") or m.get("end_line"):
                    md = {"path": m.get("path"), "start_line": m.get("start_line"), "end_line": m.get("end_line")}
                else:
                    pt = m.get("pt", {})
                    if hasattr(pt, "payload") and getattr(pt, "payload"):
                        md = (pt.payload or {}).get("metadata") or {}
        except Exception:
            md = {}
        path = str((md or {}).get("path") or "")
        start_line = int((md or {}).get("start_line") or 0)
        end_line = int((md or {}).get("end_line") or 0)
        if not path or start_line <= 0 or end_line <= 0:
            continue
        lst = clusters.setdefault(path, [])
        merged = False
        item_score = float(m.get("raw_score") or m.get("score") or m.get("s") or 0.0)
        for c in lst:
            if (
                start_line <= c["end"] + merge_lines
                and end_line >= c["start"] - merge_lines
            ):
                cluster_score = float(c["m"].get("raw_score") or c["m"].get("score") or c["m"].get("s") or 0.0)
                if item_score > cluster_score:
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged = True
                break
        if not merged:
            lst.append({"start": start_line, "end": end_line, "m": m, "p": path})

    budget = budget_tokens
    out: List[Dict[str, Any]] = []
    per_path_counts: Dict[str, int] = {}

    def _line_tokens(s: int, e: int) -> int:
        return max(1, (e - s + 1) * tokens_per_line)

    flattened = []
    for lst in clusters.values():
        for c in lst:
            flattened.append(c)

    def _flat_key(c):
        m = c.get("m", {})
        path = str(c.get("p") or "")
        start = int(c.get("start") or 0)
        score = float(m.get("raw_score") or m.get("score") or m.get("s") or 0.0)
        if score < 0:
            return (score, path, start)
        else:
            return (-score, path, start)

    flattened.sort(key=_flat_key)

    for c in flattened:
        m = c["m"]
        path = str(c.get("p") or "")
        if per_path_counts.get(path, 0) >= out_max_spans:
            continue
        need = _line_tokens(c["start"], c["end"])
        if need <= budget:
            budget -= need
            per_path_counts[path] = per_path_counts.get(path, 0) + 1
            out.append({"m": m, "start": c["start"], "end": c["end"], "need_tokens": need})
        elif budget > 0 and per_path_counts.get(path, 0) < out_max_spans:
            affordable_lines = max(1, budget // tokens_per_line)
            trim_end = c["start"] + affordable_lines - 1
            if trim_end >= c["start"]:
                trimmed_need = _line_tokens(c["start"], trim_end)
                budget -= trimmed_need
                per_path_counts[path] = per_path_counts.get(path, 0) + 1
                out.append({"m": m, "start": c["start"], "end": trim_end, "need_tokens": trimmed_need, "_trimmed": True})
        if budget <= 0:
            break

    result: List[Dict[str, Any]] = []
    for c in out:
        m = c["m"]
        m["start_line"] = c["start"]
        m["end_line"] = c["end"]
        m["text"] = ""
        m["_merged_start"] = c["start"]
        m["_merged_end"] = c["end"]
        m["_budget_tokens"] = c["need_tokens"]
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def _detect_implementation_intent(queries: List[str]) -> bool:
    """Detect if query signals user wants implementation code."""
    if not queries:
        return False
    joined = " ".join(queries).lower()
    for pattern in _IMPL_INTENT_PATTERNS:
        if pattern in joined:
            return True
    return False
