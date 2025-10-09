#!/usr/bin/env python3
import os
import argparse
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import re
import json


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
SYMBOL_EQUALITY_BOOST = float(os.environ.get("HYBRID_SYMBOL_EQUALITY_BOOST", "0.25") or 0.25)
VENDOR_PENALTY = float(os.environ.get("HYBRID_VENDOR_PENALTY", "0.05") or 0.05)
LANG_MATCH_BOOST = float(os.environ.get("HYBRID_LANG_MATCH_BOOST", "0.05") or 0.05)
CLUSTER_LINES = int(os.environ.get("HYBRID_CLUSTER_LINES", "15") or 15)

# Core file patterns (prioritize implementation over tests/docs)
CORE_FILE_PATTERNS = [
    r"\.py$", r"\.js$", r"\.ts$", r"\.tsx$", r"\.jsx$", r"\.go$", r"\.rs$", r"\.java$", r"\.cpp$", r"\.c$", r"\.h$"
]
NON_CORE_PATTERNS = [
    r"test", r"spec", r"__test__", r"\.test\.", r"\.spec\.", r"_test\.py$", r"_spec\.py$",
    r"docs?/", r"documentation/", r"\.md$", r"\.txt$", r"README", r"CHANGELOG"
]

def is_core_file(path: str) -> bool:
    """Check if file is core implementation (not test/doc)"""
    import re
    path_lower = path.lower()
    # Skip non-core files
    for pattern in NON_CORE_PATTERNS:
        if re.search(pattern, path_lower):
            return False
    # Check for core file extensions
    for pattern in CORE_FILE_PATTERNS:
        if re.search(pattern, path_lower):
            return True
    return False

# Vendor/third-party detection
VENDOR_PATTERNS = [
    "vendor/", "third_party/", "node_modules/", "/dist/", "/build/", ".generated/", "generated/", "autogen/", "target/"
]

def is_vendor_path(path: str) -> bool:
    p = path.lower()
    return any(s in p for s in VENDOR_PATTERNS)

# Language extension mapping and checks
LANG_EXTS: Dict[str, List[str]] = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
    "c": [".c", ".h"],
}

def lang_matches_path(lang: str, path: str) -> bool:
    if not lang:
        return False
    exts = LANG_EXTS.get(lang.lower(), [])
    pl = path.lower()
    return any(pl.endswith(ext) for ext in exts)

# --- Query DSL parsing (lang:, file:/path, path:, under:, kind:, symbol:) ---
def parse_query_dsl(queries: List[str]) -> Tuple[List[str], Dict[str, str]]:
    clean: List[str] = []
    extracted: Dict[str, str] = {}
    token_re = re.compile(r"\b(?:(lang|language|file|path|under|kind|symbol))\s*:\s*([^\s]+)", re.IGNORECASE)
    for q in queries:
        parts = []
        last = 0
        for m in token_re.finditer(q):
            key = m.group(1).lower()
            val = m.group(2)
            if key in ("file", "path"):
                extracted["under"] = val
            elif key in ("lang", "language"):
                extracted["language"] = val
            else:
                extracted[key] = val
            parts.append(q[last:m.start()].strip())
            last = m.end()
        parts.append(q[last:].strip())
        remaining = " ".join([p for p in parts if p])
        if remaining:
            clean.append(remaining)
    # Keep at least an empty query if everything was tokens
    if not clean and queries:
        clean = [""]
    return clean, extracted

# --- Tokenization helpers for smarter lexical ---
_STOP = {"the","a","an","of","in","on","for","and","or","to","with","by","is","are","be","this","that"}

def _split_ident(s: str) -> List[str]:
    # split snake_case and camelCase
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
    toks: List[str] = []
    for ph in phrases:
        toks.extend(_split_ident(ph))
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

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
    """Smarter lexical: split identifiers, weight matches in symbol/path higher."""
    tokens = tokenize_queries(phrases)
    if not tokens:
        return 0.0
    path = str(md.get("path", "")).lower()
    path_segs = re.split(r"[/\\]", path)
    path_text = " ".join(path_segs)
    sym = str(md.get("symbol", "")).lower()
    symp = str(md.get("symbol_path", "")).lower()
    code = str(md.get("code", ""))[:2000].lower()

    s = 0.0
    for t in tokens:
        if not t:
            continue
        if t and (t in sym or t in symp):
            s += 1.2  # symbol emphasis
        if t and any(t in seg for seg in path_segs):
            s += 0.6  # path segment
        if t and t in code:
            s += 1.0  # body occurrence
    return s


def dense_query(client: QdrantClient, vec_name: str, v: List[float], flt, per_query: int) -> List[Any]:
    try:
        ef = max(EF_SEARCH, 32 + 4*int(per_query))
        qp = client.query_points(
            collection_name=COLLECTION,
            query=v,
            using=vec_name,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
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
    ap.add_argument("--json", dest="json", action="store_true", help="Emit JSON lines with score breakdown")

    args = ap.parse_args()

    model = TextEmbedding(model_name=MODEL_NAME)
    vec_name = _sanitize_vector_name(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    # Parse Query DSL from queries, then build effective filters
    raw_queries = list(args.query)
    clean_queries, dsl = parse_query_dsl(raw_queries)
    eff_language = args.language or dsl.get("language")
    eff_under = args.under or dsl.get("under")
    eff_kind = args.kind or dsl.get("kind")
    eff_symbol = args.symbol or dsl.get("symbol")

    # Build optional filter
    flt = None
    must = []
    if eff_language:
        must.append(models.FieldCondition(key="metadata.language", match=models.MatchValue(value=eff_language)))
    if eff_under:
        must.append(models.FieldCondition(key="metadata.path_prefix", match=models.MatchValue(value=eff_under)))
    if eff_kind:
        must.append(models.FieldCondition(key="metadata.kind", match=models.MatchValue(value=eff_kind)))
    if eff_symbol:
        must.append(models.FieldCondition(key="metadata.symbol", match=models.MatchValue(value=eff_symbol)))
    flt = models.Filter(must=must) if must else None

    # Build query set (optionally expanded)
    queries = list(clean_queries)
    if args.expand:
        queries = expand_queries(queries, eff_language)

    embedded = [vec.tolist() for vec in model.embed(queries)]
    result_sets: List[List[Any]] = [dense_query(client, vec_name, v, flt, args.per_query) for v in embedded]

    # RRF fusion (weighted)
    score_map: Dict[str, Dict[str, Any]] = {}
    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(pid, {"pt": p, "s": 0.0, "d": 0.0, "lx": 0.0, "sym_sub": 0.0, "sym_eq": 0.0, "core": 0.0, "vendor": 0.0, "langb": 0.0, "rec": 0.0})
            dens = DENSE_WEIGHT * rrf(rank)
            score_map[pid]["d"] += dens
            score_map[pid]["s"] += dens

    # Lexical bump + symbol boost; also collect recency
    timestamps: List[int] = []
    for pid, rec in list(score_map.items()):
        md = (rec["pt"].payload or {}).get("metadata") or {}
        lx = LEXICAL_WEIGHT * lexical_score(queries, md)
        rec["lx"] += lx
        rec["s"] += lx
        ts = md.get("ingested_at")
        if isinstance(ts, int):
            timestamps.append(ts)

        # Symbol-based boosts
        sym = str(md.get("symbol") or "").lower()
        sym_path = str(md.get("symbol_path") or "").lower()
        sym_text = f"{sym} {sym_path}"
        for q in queries:
            ql = q.lower()
            if not ql:
                continue
            # substring match boost
            if ql in sym_text:
                rec["sym_sub"] += SYMBOL_BOOST
                rec["s"] += SYMBOL_BOOST
            # exact match boost (symbol or symbol_path)
            if ql == sym or ql == sym_path:
                rec["sym_eq"] += SYMBOL_EQUALITY_BOOST
                rec["s"] += SYMBOL_EQUALITY_BOOST

        # Path-based adjustments
        path = str(md.get("path") or "")
        if CORE_FILE_BOOST > 0.0 and path and is_core_file(path):
            rec["core"] += CORE_FILE_BOOST
            rec["s"] += CORE_FILE_BOOST
        if VENDOR_PENALTY > 0.0 and path and is_vendor_path(path):
            rec["vendor"] -= VENDOR_PENALTY
            rec["s"] -= VENDOR_PENALTY

        # Language match boost if requested
        if LANG_MATCH_BOOST > 0.0 and path and (eff_language or getattr(args, "language", None)):
            lang = str((eff_language or args.language or "")).lower()
            md_lang = str((md.get("language") or "")).lower()
            if (lang and md_lang and md_lang == lang) or lang_matches_path(lang, path):
                rec["langb"] += LANG_MATCH_BOOST
                rec["s"] += LANG_MATCH_BOOST

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

    # Rank with deterministic tie-breakers
    def _tie_key(m: Dict[str, Any]):
        md = (m["pt"].payload or {}).get("metadata") or {}
        sp = str(md.get("symbol_path") or md.get("symbol") or "")
        path = str(md.get("path") or "")
        start_line = int(md.get("start_line") or 0)
        return (-float(m["s"]), len(sp), path, start_line)

    ranked = sorted(score_map.values(), key=_tie_key)

    # Adjacent-hit clustering by path
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for m in ranked:
        md = (m["pt"].payload or {}).get("metadata") or {}
        path = str(md.get("path") or "")
        start_line = int(md.get("start_line") or 0)
        end_line = int(md.get("end_line") or 0)
        lst = clusters.setdefault(path, [])
        merged_flag = False
        for c in lst:
            if start_line <= c["end"] + CLUSTER_LINES and end_line >= c["start"] - CLUSTER_LINES:
                # Near/overlapping: keep the higher-scoring rep and expand bounds
                if float(m["s"]) > float(c["m"]["s"]):
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged_flag = True
                break
        if not merged_flag:
            lst.append({"start": start_line, "end": end_line, "m": m})

    ranked = sorted([c["m"] for lst in clusters.values() for c in lst], key=_tie_key)

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

