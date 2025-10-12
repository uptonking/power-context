#!/usr/bin/env python3
import os
import argparse
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import re
import json


def _collection() -> str:
    return os.environ.get("COLLECTION_NAME", "my-collection")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")


LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)


# Ensure project root is on sys.path when run as a script (so 'scripts' package imports work)
import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils import sanitize_vector_name as _sanitize_vector_name


RRF_K = int(os.environ.get("HYBRID_RRF_K", "60") or 60)
DENSE_WEIGHT = float(os.environ.get("HYBRID_DENSE_WEIGHT", "1.0") or 1.0)
LEXICAL_WEIGHT = float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "0.25") or 0.25)
LEX_VECTOR_WEIGHT = float(os.environ.get("HYBRID_LEX_VECTOR_WEIGHT", str(LEXICAL_WEIGHT)) or LEXICAL_WEIGHT)
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
    token_re = re.compile(r"\b(?:(lang|language|file|path|under|kind|symbol|ext|not|case|repo))\s*:\s*([^\s]+)", re.IGNORECASE)
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
            elif key in ("ext",):
                extracted["ext"] = val
            elif key in ("not",):
                extracted["not"] = val
            elif key in ("case",):
                extracted["case"] = val
            elif key in ("repo",):
                extracted["repo"] = val
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
        if t in sym or t in symp:
            s += 2.0
        if any(t in seg for seg in path_segs):
            s += 0.6
        if t in code:
            s += 1.0
    return s



# --- Lexical vector (hashing trick) for server-side hybrid ---
def _split_ident_lex(s: str) -> List[str]:
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def lex_hash_vector(phrases: List[str], dim: int = LEX_VECTOR_DIM) -> List[float]:
    import hashlib, math
    toks: List[str] = []
    for ph in phrases:
        toks.extend(_split_ident_lex(ph))
    if not toks:
        return [0.0] * dim
    vec = [0.0] * dim
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec]


def lex_query(client: QdrantClient, v: List[float], flt, per_query: int) -> List[Any]:
    try:
        ef = max(EF_SEARCH, 32 + 4*int(per_query))
        qp = client.query_points(
            collection_name=_collection(),
            query=v,
            using=LEX_VECTOR_NAME,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except Exception:
        res = client.search(
            collection_name=_collection(),
            query_vector={"name": LEX_VECTOR_NAME, "vector": v},
            limit=per_query,
            with_payload=True,
            query_filter=flt,
        )
        return res


def dense_query(client: QdrantClient, vec_name: str, v: List[float], flt, per_query: int) -> List[Any]:
    try:
        ef = max(EF_SEARCH, 32 + 4*int(per_query))
        qp = client.query_points(
            collection_name=_collection(),
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
            collection_name=_collection(),
            query_vector={"name": vec_name, "vector": v},
            limit=per_query,
            with_payload=True,
            query_filter=flt,
        )
        return res


# In-process API: run hybrid search and return structured items list
# Optional: pass an existing TextEmbedding instance via model to reuse cache

def run_hybrid_search(
    queries: List[str],
    limit: int = 10,
    per_path: int = 1,
    language: str | None = None,
    under: str | None = None,
    kind: str | None = None,
    symbol: str | None = None,
    ext: str | None = None,
    not_filter: str | None = None,
    case: str | None = None,
    path_regex: str | None = None,
    path_glob: str | list[str] | None = None,
    not_glob: str | list[str] | None = None,
    expand: bool = True,
    model: TextEmbedding | None = None,
) -> List[Dict[str, Any]]:
    client = QdrantClient(url=os.environ.get("QDRANT_URL", QDRANT_URL), api_key=API_KEY)
    model_name = os.environ.get("EMBEDDING_MODEL", MODEL_NAME)
    _model = model or TextEmbedding(model_name=model_name)
    vec_name = _sanitize_vector_name(model_name)

    # Parse Query DSL and merge with explicit args
    raw_queries = list(queries)
    clean_queries, dsl = parse_query_dsl(raw_queries)
    eff_language = language or dsl.get("language")
    eff_under = under or dsl.get("under")
    eff_kind = kind or dsl.get("kind")
    eff_symbol = symbol or dsl.get("symbol")
    eff_ext = ext or dsl.get("ext")
    eff_not = not_filter or dsl.get("not")
    eff_case = case or dsl.get("case") or os.environ.get("HYBRID_CASE", "insensitive")
    eff_repo = dsl.get("repo")
    eff_path_regex = path_regex
    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            out = []
            for e in x:
                s = str(e).strip()
                if s:
                    out.append(s)
            return out
        s = str(x).strip()
        return [s] if s else []
    eff_path_globs = _to_list(path_glob)
    eff_not_globs = _to_list(not_glob)

    # Normalize under
    def _norm_under(u: str | None) -> str | None:
        if not u:
            return None
        u = str(u).strip().replace("\\", "/")
        u = "/".join([p for p in u.split("/") if p])
        if not u:
            return None
        if not u.startswith("/"):
            v = "/work/" + u
        else:
            v = "/work/" + u.lstrip("/") if not u.startswith("/work/") else u
        return v
    eff_under = _norm_under(eff_under)

    # Build optional filter
    flt = None
    must = []
    if eff_language:
        must.append(models.FieldCondition(key="metadata.language", match=models.MatchValue(value=eff_language)))
    if eff_repo:
        must.append(models.FieldCondition(key="metadata.repo", match=models.MatchValue(value=eff_repo)))
    if eff_under:
        must.append(models.FieldCondition(key="metadata.path_prefix", match=models.MatchValue(value=eff_under)))
    if eff_kind:
        must.append(models.FieldCondition(key="metadata.kind", match=models.MatchValue(value=eff_kind)))
    if eff_symbol:
        must.append(models.FieldCondition(key="metadata.symbol", match=models.MatchValue(value=eff_symbol)))
    flt = models.Filter(must=must) if must else None

    # Optionally expand queries
    qlist = list(clean_queries)
    if expand:
        qlist = expand_queries(qlist, eff_language)

    # Lexical vector query
    score_map: Dict[str, Dict[str, Any]] = {}
    try:
        lex_vec = lex_hash_vector(qlist)
        lex_results = lex_query(client, lex_vec, flt, max(24, limit))
    except Exception:
        lex_results = []
    for rank, p in enumerate(lex_results, 1):
        pid = str(p.id)
        score_map.setdefault(pid, {"pt": p, "s": 0.0, "d": 0.0, "lx": 0.0, "sym_sub": 0.0, "sym_eq": 0.0, "core": 0.0, "vendor": 0.0, "langb": 0.0, "rec": 0.0})
        lxs = LEX_VECTOR_WEIGHT * rrf(rank)
        score_map[pid]["lx"] += lxs
        score_map[pid]["s"] += lxs

    # Dense queries
    embedded = [vec.tolist() for vec in _model.embed(qlist)]
    result_sets: List[List[Any]] = [dense_query(client, vec_name, v, flt, max(24, limit)) for v in embedded]
    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(pid, {"pt": p, "s": 0.0, "d": 0.0, "lx": 0.0, "sym_sub": 0.0, "sym_eq": 0.0, "core": 0.0, "vendor": 0.0, "langb": 0.0, "rec": 0.0})
            dens = DENSE_WEIGHT * rrf(rank)
            score_map[pid]["d"] += dens
            score_map[pid]["s"] += dens

    # Lexical + boosts
    timestamps: List[int] = []
    for pid, rec in list(score_map.items()):
        md = (rec["pt"].payload or {}).get("metadata") or {}
        lx = LEXICAL_WEIGHT * lexical_score(qlist, md)
        rec["lx"] += lx
        rec["s"] += lx
        ts = md.get("last_modified_at") or md.get("ingested_at")
        if isinstance(ts, int):
            timestamps.append(ts)
        sym = str(md.get("symbol") or "").lower()
        sym_path = str(md.get("symbol_path") or "").lower()
        sym_text = f"{sym} {sym_path}"
        for q in qlist:
            ql = q.lower()
            if not ql:
                continue
            if ql in sym_text:
                rec["sym_sub"] += SYMBOL_BOOST
                rec["s"] += SYMBOL_BOOST
            if ql == sym or ql == sym_path:
                rec["sym_eq"] += SYMBOL_EQUALITY_BOOST
                rec["s"] += SYMBOL_EQUALITY_BOOST
        path = str(md.get("path") or "")
        if CORE_FILE_BOOST > 0.0 and path and is_core_file(path):
            rec["core"] += CORE_FILE_BOOST
            rec["s"] += CORE_FILE_BOOST
        if VENDOR_PENALTY > 0.0 and path and is_vendor_path(path):
            rec["vendor"] -= VENDOR_PENALTY
            rec["s"] -= VENDOR_PENALTY
        if LANG_MATCH_BOOST > 0.0 and path and eff_language:
            lang = str(eff_language).lower()
            md_lang = str((md.get("language") or "")).lower()
            if (lang and md_lang and md_lang == lang) or lang_matches_path(lang, path):
                rec["langb"] += LANG_MATCH_BOOST
                rec["s"] += LANG_MATCH_BOOST

    if timestamps and RECENCY_WEIGHT > 0.0:
        tmin, tmax = min(timestamps), max(timestamps)
        span = max(1, tmax - tmin)
        for rec in score_map.values():
            md = (rec["pt"].payload or {}).get("metadata") or {}
            ts = md.get("last_modified_at") or md.get("ingested_at")
            if isinstance(ts, int):
                norm = (ts - tmin) / span
                rec_comp = RECENCY_WEIGHT * norm
                rec["rec"] += rec_comp
                rec["s"] += rec_comp

    def _tie_key(m: Dict[str, Any]):
        md = (m["pt"].payload or {}).get("metadata") or {}
        sp = str(md.get("symbol_path") or md.get("symbol") or "")
        path = str(md.get("path") or "")
        start_line = int(md.get("start_line") or 0)
        return (-float(m["s"]), len(sp), path, start_line)

    ranked = sorted(score_map.values(), key=_tie_key)

    # Cluster by path adjacency
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
                if float(m["s"]) > float(c["m"]["s"]):
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged_flag = True
                break
        if not merged_flag:
            lst.append({"start": start_line, "end": end_line, "m": m})

    ranked = sorted([c["m"] for lst in clusters.values() for c in lst], key=_tie_key)

    # Client-side filters and per-path diversification
    import re as _re, fnmatch as _fnm
    case_sensitive = (str(eff_case or "").lower() == "sensitive")
    def _match_glob(pat: str, path: str) -> bool:
        if not pat:
            return True
        if case_sensitive:
            return _fnm.fnmatchcase(path, pat)
        return _fnm.fnmatchcase(path.lower(), pat.lower())

    if eff_not or eff_path_regex or eff_ext or eff_path_globs or eff_not_globs:
        def _pass_filters(m: Dict[str, Any]) -> bool:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path") or "")
            pp = str(md.get("path_prefix") or "")
            p_for_sub = path if case_sensitive else path.lower()
            pp_for_sub = pp if case_sensitive else pp.lower()
            if eff_not:
                nn = eff_not if case_sensitive else eff_not.lower()
                if nn in p_for_sub or nn in pp_for_sub:
                    return False
            if eff_not_globs and any(_match_glob(g, path) for g in eff_not_globs):
                return False
            if eff_ext:
                ex = eff_ext.lower().lstrip('.')
                if not path.lower().endswith('.' + ex):
                    return False
            if eff_path_regex:
                flags = 0 if case_sensitive else _re.IGNORECASE
                try:
                    if not _re.search(eff_path_regex, path, flags=flags):
                        return False
                except Exception:
                    pass
            if eff_path_globs and not any(_match_glob(g, path) for g in eff_path_globs):
                return False
            return True
        ranked = [m for m in ranked if _pass_filters(m)]

    if per_path and per_path > 0:
        counts: Dict[str, int] = {}
        merged: List[Dict[str, Any]] = []
        for m in ranked:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path", ""))
            c = counts.get(path, 0)
            if c < per_path:
                merged.append(m)
                counts[path] = c + 1
            if len(merged) >= limit:
                break
    else:
        merged = ranked[: limit]

    # Emit structured items
    items: List[Dict[str, Any]] = []
    for m in merged:
        md = (m["pt"].payload or {}).get("metadata") or {}
        comp = {
            "dense_rrf": round(float(m.get("d", 0.0)), 4),
            "lexical": round(float(m.get("lx", 0.0)), 4),
            "symbol_substr": round(float(m.get("sym_sub", 0.0)), 4),
            "symbol_exact": round(float(m.get("sym_eq", 0.0)), 4),
            "core_boost": round(float(m.get("core", 0.0)), 4),
            "vendor_penalty": round(float(m.get("vendor", 0.0)), 4),
            "lang_boost": round(float(m.get("langb", 0.0)), 4),
            "recency": round(float(m.get("rec", 0.0)), 4),
        }
        why = []
        if comp["dense_rrf"]:
            why.append(f"dense_rrf:{comp['dense_rrf']}")
        for k in ("lexical","symbol_substr","symbol_exact","core_boost","lang_boost"):
            if comp[k]:
                why.append(f"{k}:{comp[k]}")
        if comp["vendor_penalty"]:
            why.append(f"vendor_penalty:{comp['vendor_penalty']}")
        if comp["recency"]:
            why.append(f"recency:{comp['recency']}")
        items.append({
            "score": round(float(m["s"]), 4),
            "path": md.get("path"),
            "symbol": md.get("symbol_path") or md.get("symbol") or "",
            "start_line": md.get("start_line"),
            "end_line": md.get("end_line"),
            "components": comp,
            "why": why,
        })
    return items


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
    # Structured filters to mirror MCP tool fields
    ap.add_argument("--ext", type=str, default=None)
    ap.add_argument("--not", dest="not_filter", type=str, default=None)
    ap.add_argument("--case", type=str, choices=["sensitive", "insensitive"], default=os.environ.get("HYBRID_CASE", "insensitive"))
    ap.add_argument("--path-regex", dest="path_regex", type=str, default=None)
    ap.add_argument("--path-glob", dest="path_glob", type=str, default=None)
    ap.add_argument("--not-glob", dest="not_glob", type=str, default=None)

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
    eff_ext = args.ext or dsl.get("ext")
    eff_not = args.not_filter or dsl.get("not")
    eff_case = args.case or dsl.get("case")
    eff_repo = dsl.get("repo")
    eff_path_regex = args.path_regex
    eff_path_glob = getattr(args, "path_glob", None)
    eff_not_glob = getattr(args, "not_glob", None)

    # Normalize 'under' to absolute path_prefix used in payload (defaults to /work/<rel>)
    def _norm_under(u: str | None) -> str | None:
        if not u:
            return None
        u = str(u).strip()
        # Handle common path variations: backslashes, multiple slashes, trailing slashes
        u = u.replace("\\", "/")
        # Collapse multiple slashes and remove empty segments
        u = "/".join([p for p in u.split("/") if p])
        if not u:
            return None
        # Relative path: prepend /work/
        if not u.startswith("/"):
            v = "/work/" + u
        else:
            # Absolute path: ensure it's under /work mount
            v = "/work/" + u.lstrip("/") if not u.startswith("/work/") else u
        # If the normalized path points to a real file under /work, use its parent directory as prefix
        try:
            from pathlib import Path as _P
            p = _P(v)
            # If it's an existing file, use its parent directory as the prefix
            if p.is_file():
                return str(p.parent)
            # Heuristic: if path doesn't exist and looks like a file stem (no dot),
            # treat it as a file name and use its parent directory
            if (not p.exists()) and p.name and ("." not in p.name):
                return str(p.parent) if str(p.parent) else v
        except Exception:
            pass
        # Already normalized /work/... dir path or non-existent path; use as-is
        return v

    eff_under = _norm_under(eff_under)

    # Build optional filter
    flt = None
    must = []
    if eff_language:
        must.append(models.FieldCondition(key="metadata.language", match=models.MatchValue(value=eff_language)))
    if eff_repo:
        must.append(models.FieldCondition(key="metadata.repo", match=models.MatchValue(value=eff_repo)))
    if eff_under:
        must.append(models.FieldCondition(key="metadata.path_prefix", match=models.MatchValue(value=eff_under)))
    # If ext: was provided without an explicit language, infer language from extension
    if eff_ext and not eff_language:
        ex = eff_ext.lower().lstrip('.')
        for lang, exts in LANG_EXTS.items():
            if any(ex == e.lstrip('.').lower() for e in exts):
                eff_language = lang
                break

    if eff_kind:
        must.append(models.FieldCondition(key="metadata.kind", match=models.MatchValue(value=eff_kind)))
    if eff_symbol:
        must.append(models.FieldCondition(key="metadata.symbol", match=models.MatchValue(value=eff_symbol)))
    flt = models.Filter(must=must) if must else None

    # Build query set (optionally expanded)
    queries = list(clean_queries)
    # Initialize score map early so we can accumulate from lex and dense
    score_map: Dict[str, Dict[str, Any]] = {}
    # Server-side lexical vector search (hashing) as an additional ranked list
    try:
        lex_vec = lex_hash_vector(queries)
        lex_results = lex_query(client, lex_vec, flt, args.per_query)
    except Exception:
        lex_results = []

    if args.expand:
        queries = expand_queries(queries, eff_language)

    # Add server-side lexical vector ranking into fusion
    for rank, p in enumerate(lex_results, 1):
        pid = str(p.id)
        score_map.setdefault(pid, {"pt": p, "s": 0.0, "d": 0.0, "lx": 0.0, "sym_sub": 0.0, "sym_eq": 0.0, "core": 0.0, "vendor": 0.0, "langb": 0.0, "rec": 0.0})
        lxs = LEX_VECTOR_WEIGHT * rrf(rank)
        score_map[pid]["lx"] += lxs
        score_map[pid]["s"] += lxs

    embedded = [vec.tolist() for vec in model.embed(queries)]
    result_sets: List[List[Any]] = [dense_query(client, vec_name, v, flt, args.per_query) for v in embedded]

    # RRF fusion (weighted)
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
        ts = md.get("last_modified_at") or md.get("ingested_at")
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
            ts = md.get("last_modified_at") or md.get("ingested_at")
            if isinstance(ts, int):
                norm = (ts - tmin) / span
                rec_comp = RECENCY_WEIGHT * norm
                rec["rec"] += rec_comp
                rec["s"] += rec_comp
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

    # Apply client-side filters: NOT substring, path regex, glob, and ext
    import re as _re, fnmatch as _fnm
    case_sensitive = (str(eff_case or "").lower() == "sensitive")
    if eff_not or eff_path_regex or eff_ext or eff_path_glob or eff_not_glob:
        def _match_glob(pat: str, path: str) -> bool:
            if not pat:
                return True
            if case_sensitive:
                return _fnm.fnmatchcase(path, pat)
            return _fnm.fnmatchcase(path.lower(), pat.lower())
        def _pass_filters(m: Dict[str, Any]) -> bool:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path") or "")
            pp = str(md.get("path_prefix") or "")
            p_for_sub = path if case_sensitive else path.lower()
            pp_for_sub = pp if case_sensitive else pp.lower()
            # NOT substring filter
            if eff_not:
                nn = eff_not if case_sensitive else eff_not.lower()
                if nn in p_for_sub or nn in pp_for_sub:
                    return False
            # not_glob exclusion
            if eff_not_glob and _match_glob(eff_not_glob, path):
                return False
            # Extension filter (normalize to .ext)
            if eff_ext:
                ex = eff_ext.lower().lstrip('.')
                if not path.lower().endswith('.' + ex):
                    return False
            # Path regex filter
            if eff_path_regex:
                flags = 0 if case_sensitive else _re.IGNORECASE
                try:
                    if not _re.search(eff_path_regex, path, flags=flags):
                        return False
                except Exception:
                    # Ignore invalid regex
                    pass
            # path_glob inclusion
            if eff_path_glob and not _match_glob(eff_path_glob, path):
                return False
            return True
        ranked = [m for m in ranked if _pass_filters(m)]

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
        if getattr(args, "json", False):
            item = {
                "score": round(float(m["s"]), 4),
                "path": md.get("path"),
                "symbol": md.get("symbol_path") or md.get("symbol") or "",
                "start_line": md.get("start_line"),
                "end_line": md.get("end_line"),
                "components": {
                    "dense_rrf": round(float(m.get("d", 0.0)), 4),
                    "lexical": round(float(m.get("lx", 0.0)), 4),
                    "symbol_substr": round(float(m.get("sym_sub", 0.0)), 4),
                    "symbol_exact": round(float(m.get("sym_eq", 0.0)), 4),
                    "core_boost": round(float(m.get("core", 0.0)), 4),
                    "vendor_penalty": round(float(m.get("vendor", 0.0)), 4),
                    "lang_boost": round(float(m.get("langb", 0.0)), 4),
                    "recency": round(float(m.get("rec", 0.0)), 4),
                },
            }
            # Build a human friendly why list
            why = []
            if item["components"]["dense_rrf"]:
                why.append(f"dense_rrf:{item['components']['dense_rrf']}")
            for k in ("lexical","symbol_substr","symbol_exact","core_boost","lang_boost"):
                if item["components"][k]:
                    why.append(f"{k}:{item['components'][k]}")
            if item["components"]["vendor_penalty"]:
                why.append(f"vendor_penalty:{item['components']['vendor_penalty']}")
            if item["components"]["recency"]:
                why.append(f"recency:{item['components']['recency']}")
            item["why"] = why
            print(json.dumps(item))
        else:
            print(f"{m['s']:.3f}\t{md.get('path')}\t{md.get('symbol_path') or md.get('symbol') or ''}\t{md.get('start_line')}-{md.get('end_line')}")


if __name__ == "__main__":
    main()

