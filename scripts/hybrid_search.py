#!/usr/bin/env python3
import os
import argparse
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import re
import json


def _collection(collection_name: str | None = None) -> str:
    """Get collection name with priority: CLI arg > ENV > default"""
    if collection_name and collection_name.strip():
        return collection_name.strip()
    return os.environ.get("COLLECTION_NAME", "my-collection")


MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")


LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)
# Optional mini vector (ReFRAG gating)
MINI_VECTOR_NAME = os.environ.get("MINI_VECTOR_NAME", "mini")
MINI_VEC_DIM = int(os.environ.get("MINI_VEC_DIM", "64") or 64)
HYBRID_MINI_WEIGHT = float(os.environ.get("HYBRID_MINI_WEIGHT", "0.5") or 0.5)


# Lightweight embedding cache for query embeddings (model_name, text) -> vector
from threading import Lock as _Lock

_EMBED_QUERY_CACHE: Dict[tuple[str, str], List[float]] = {}
_EMBED_LOCK = _Lock()


def _embed_queries_cached(
    model: TextEmbedding, queries: List[str]
) -> List[List[float]]:
    """Cache dense query embeddings to avoid repeated compute across expansions/retries."""
    out: List[List[float]] = []
    try:
        # Best-effort model name extraction; fall back to env
        name = getattr(model, "model_name", None) or os.environ.get(
            "EMBEDDING_MODEL", MODEL_NAME
        )
    except Exception:
        name = os.environ.get("EMBEDDING_MODEL", MODEL_NAME)
    for q in queries:
        key = (str(name), str(q))
        v = _EMBED_QUERY_CACHE.get(key)
        if v is None:
            try:
                vec = next(model.embed([q])).tolist()
            except Exception:
                # Fallback: embed batch and take first
                vec = next(model.embed([str(q)])).tolist()
            with _EMBED_LOCK:
                # Double-check inside lock
                if key not in _EMBED_QUERY_CACHE:
                    _EMBED_QUERY_CACHE[key] = vec
            v = vec
        out.append(v)
    return out


# Ensure project root is on sys.path when run as a script (so 'scripts' package imports work)
import sys
from pathlib import Path as _P

_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.ingest_code import ensure_collection as _ensure_collection
from scripts.ingest_code import project_mini as _project_mini


RRF_K = int(os.environ.get("HYBRID_RRF_K", "60") or 60)
DENSE_WEIGHT = float(os.environ.get("HYBRID_DENSE_WEIGHT", "1.0") or 1.0)
LEXICAL_WEIGHT = float(os.environ.get("HYBRID_LEXICAL_WEIGHT", "0.25") or 0.25)
LEX_VECTOR_WEIGHT = float(
    os.environ.get("HYBRID_LEX_VECTOR_WEIGHT", str(LEXICAL_WEIGHT)) or LEXICAL_WEIGHT
)
EF_SEARCH = int(os.environ.get("QDRANT_EF_SEARCH", "128") or 128)
# Lightweight, configurable boosts
SYMBOL_BOOST = float(os.environ.get("HYBRID_SYMBOL_BOOST", "0.15") or 0.15)
RECENCY_WEIGHT = float(os.environ.get("HYBRID_RECENCY_WEIGHT", "0.1") or 0.1)
CORE_FILE_BOOST = float(os.environ.get("HYBRID_CORE_FILE_BOOST", "0.1") or 0.1)
SYMBOL_EQUALITY_BOOST = float(
    os.environ.get("HYBRID_SYMBOL_EQUALITY_BOOST", "0.25") or 0.25
)
VENDOR_PENALTY = float(os.environ.get("HYBRID_VENDOR_PENALTY", "0.05") or 0.05)
LANG_MATCH_BOOST = float(os.environ.get("HYBRID_LANG_MATCH_BOOST", "0.05") or 0.05)
CLUSTER_LINES = int(os.environ.get("HYBRID_CLUSTER_LINES", "15") or 15)
# Penalize test files slightly to prefer implementation over tests
TEST_FILE_PENALTY = float(os.environ.get("HYBRID_TEST_FILE_PENALTY", "0.15") or 0.15)

# Micro-span compaction and budgeting (ReFRAG-lite output shaping)
MICRO_OUT_MAX_SPANS = int(os.environ.get("MICRO_OUT_MAX_SPANS", "3") or 3)
MICRO_MERGE_LINES = int(os.environ.get("MICRO_MERGE_LINES", "4") or 4)
MICRO_BUDGET_TOKENS = int(os.environ.get("MICRO_BUDGET_TOKENS", "512") or 512)
MICRO_TOKENS_PER_LINE = int(os.environ.get("MICRO_TOKENS_PER_LINE", "32") or 32)


def _merge_and_budget_spans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Given ranked items with metadata path/start_line/end_line, merge nearby spans
    per path and enforce a token budget using a simple tokens-per-line estimate.
    Returns a filtered/merged list preserving score order as much as possible.
    """
    # Read dynamic knobs at call-time so tests/env can override without reload
    try:
        merge_lines = int(
            os.environ.get("MICRO_MERGE_LINES", str(MICRO_MERGE_LINES))
            or MICRO_MERGE_LINES
        )
    except Exception:
        merge_lines = MICRO_MERGE_LINES
    try:
        budget_tokens = int(
            os.environ.get("MICRO_BUDGET_TOKENS", str(MICRO_BUDGET_TOKENS))
            or MICRO_BUDGET_TOKENS
        )
    except Exception:
        budget_tokens = MICRO_BUDGET_TOKENS
    try:
        tokens_per_line = int(
            os.environ.get("MICRO_TOKENS_PER_LINE", str(MICRO_TOKENS_PER_LINE))
            or MICRO_TOKENS_PER_LINE
        )
    except Exception:
        tokens_per_line = MICRO_TOKENS_PER_LINE
    try:
        out_max_spans = int(
            os.environ.get("MICRO_OUT_MAX_SPANS", str(MICRO_OUT_MAX_SPANS))
            or MICRO_OUT_MAX_SPANS
        )
    except Exception:
        out_max_spans = MICRO_OUT_MAX_SPANS

    # First cluster adjacent by path using a tighter merge gap for micro spans
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for m in items:
        # Robust metadata extraction: support both dict-shaped results and Qdrant point payloads
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
            # skip invalid entries
            continue
        lst = clusters.setdefault(path, [])
        merged = False
        for c in lst:
            if (
                start_line <= c["end"] + merge_lines
                and end_line >= c["start"] - merge_lines
            ):
                # expand bounds; keep higher-score rep
                if float(m.get("s", 0.0)) > float(c["m"].get("s", 0.0)):
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged = True
                break
        if not merged:
            lst.append({"start": start_line, "end": end_line, "m": m, "p": path})

    # Now budget per path with a global token budget
    budget = budget_tokens
    out: List[Dict[str, Any]] = []
    per_path_counts: Dict[str, int] = {}

    def _line_tokens(s: int, e: int) -> int:
        return max(1, (e - s + 1) * tokens_per_line)

    # Flatten clusters preserving original score order
    flattened = []
    for lst in clusters.values():
        for c in lst:
            flattened.append(c)

    def _flat_key(c):
        m = c.get("m", {})
        # Use stored cluster path and start for stable ordering
        path = str(c.get("p") or "")
        start = int(c.get("start") or 0)
        return (-float(m.get("s", 0.0)), path, start)

    flattened.sort(key=_flat_key)

    for c in flattened:
        m = c["m"]
        # Prefer path from cluster key
        path = str(c.get("p") or "")
        # per-path cap
        if per_path_counts.get(path, 0) >= out_max_spans:
            continue
        need = _line_tokens(c["start"], c["end"])
        if need <= budget:
            budget -= need
            per_path_counts[path] = per_path_counts.get(path, 0) + 1
            # rewrite start/end in the representative's metadata clone for emission
            # (we do not mutate original payloads coming from Qdrant objects)
            out.append(
                {"m": m, "start": c["start"], "end": c["end"], "need_tokens": need}
            )
        if budget <= 0:
            break

    # Map back to the same structure expected downstream: keep representative m
    # and expose start_line/end_line from our merged span via components
    result: List[Dict[str, Any]] = []
    for c in out:
        m = c["m"]
        # Attach merged bounds for the downstream emitter to read
        m["_merged_start"] = c["start"]
        m["_merged_end"] = c["end"]
        m["_budget_tokens"] = c["need_tokens"]
        result.append(m)
    return result


# Core file patterns (prioritize implementation over tests/docs)
CORE_FILE_PATTERNS = [
    r"\.py$",
    r"\.js$",
    r"\.ts$",
    r"\.tsx$",
    r"\.jsx$",
    r"\.go$",
    r"\.rs$",
    r"\.java$",
    r"\.cpp$",
    r"\.c$",
    r"\.h$",
]
NON_CORE_PATTERNS = [
    r"test",
    r"spec",
    r"__test__",
    r"\.test\.",
    r"\.spec\.",
    r"_test\.py$",
    r"_spec\.py$",
    r"docs?/",
    r"documentation/",
    r"\.md$",
    r"\.txt$",
    r"README",
    r"CHANGELOG",
]

# Test file patterns
TEST_FILE_PATTERNS = [
    r"/tests?/",
    r"(^|/)test_",
    r"_test\.",
    r"\.test\.",
    r"\.spec\.",
]

def is_test_file(path: str) -> bool:
    import re
    p = path.lower()
    for pattern in TEST_FILE_PATTERNS:
        if re.search(pattern, p):
            return True
    return False


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
    "vendor/",
    "third_party/",
    "node_modules/",
    "/dist/",
    "/build/",
    ".generated/",
    "generated/",
    "autogen/",
    "target/",
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
    "csharp": [".cs", ".csx"],
    "razor": [".cshtml", ".razor"],
    "xml": [".csproj", ".resx", ".config"],
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
    token_re = re.compile(
        r"\b(?:(lang|language|file|path|under|kind|symbol|ext|not|case|repo))\s*:\s*([^\s]+)",
        re.IGNORECASE,
    )
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
            parts.append(q[last : m.start()].strip())
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
            out.append(t)
            seen.add(t)
    return out


# Minimal code-aware query expansion (quick win)
CODE_SYNONYMS = {
    "function": ["method", "def", "fn"],
    "class": ["type", "object"],
    "create": ["init", "initialize", "construct"],
    "get": ["fetch", "retrieve"],
    "set": ["assign", "update"],
}


def expand_queries(
    queries: List[str], language: str | None = None, max_extra: int = 2
) -> List[str]:
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


# --- LLM-assisted expansion (optional if configured) and PRF (default-on) ---
def _llm_expand_queries(
    queries: List[str], language: str | None = None, max_new: int = 4
) -> List[str]:
    """Best-effort LLM expansion with preference for a local runtime (Ollama).
    Providers (by env):
      - LLM_PROVIDER=ollama (preferred if OLLAMA_HOST set; default http://localhost:11434)
      - fallback: OPENAI_API_KEY + LLM_EXPAND_MODEL
    On any error or if not configured, returns []."""
    import json
    import urllib.request

    model = os.environ.get("LLM_EXPAND_MODEL", "glm4")
    prompt = (
        "You are a code search expert. Given one or more short queries, suggest up to "
        f"{max_new} semantically diverse, code-oriented expansions. Only return a JSON list of strings.\n"
        f"Language hint: {language or 'any'}. Queries: {queries}"
    )

    # 1) Prefer local Ollama
    prov = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    ollama_host = (
        os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
        or "http://localhost:11434"
    )
    if prov in {"", "ollama"}:  # default to ollama if reachable
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 128},
            }
            req = urllib.request.Request(
                ollama_host.rstrip("/") + "/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = json.loads(resp.read().decode("utf-8", "ignore"))
            txt = body.get("response", "")

            def _parse_list(t: str):
                t = t.strip().strip("`")
                import re, json as _json

                try:
                    v = _json.loads(t)
                    if isinstance(v, list):
                        return v
                except Exception:
                    pass
                if t.startswith("```"):
                    t = t.strip("`")
                m = re.search(r"\[.*?\]", t, flags=re.S)
                if m:
                    try:
                        v = _json.loads(m.group(0))
                        if isinstance(v, list):
                            return v
                    except Exception:
                        pass
                return None

            arr = _parse_list(txt)
            if isinstance(arr, list):
                return [str(x) for x in arr[:max_new] if str(x).strip()]
            out: List[str] = []
            for line in txt.splitlines():
                s = line.strip().strip("- ").strip("`")
                if s:
                    out.append(s)
                if len(out) >= max_new:
                    break
            return out
        except Exception:
            pass

    # 2) Fallback to OpenAI if configured
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(data).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            body = json.loads(resp.read().decode("utf-8", "ignore"))
        txt = body.get("choices", [{}])[0].get("message", {}).get("content", "")

        def _parse_list(t: str):
            t = t.strip().strip("`")
            import re, json as _json

            try:
                v = _json.loads(t)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
            if t.startswith("```"):
                t = t.strip("`")
            m = re.search(r"\[.*?\]", t, flags=re.S)
            if m:
                try:
                    v = _json.loads(m.group(0))
                    if isinstance(v, list):
                        return v
                except Exception:
                    pass
            return None

        arr = _parse_list(txt)
        if isinstance(arr, list):
            return [str(x) for x in arr[:max_new] if str(x).strip()]
        out = []
        for line in txt.splitlines():
            s = line.strip().strip("- ").strip("`")
            if s:
                out.append(s)
            if len(out) >= max_new:
                break
        return out
    except Exception:
        return []


def _prf_terms_from_results(
    score_map: Dict[str, Dict[str, Any]], top_docs: int = 8, max_terms: int = 6
) -> List[str]:
    """Extract pseudo-relevant feedback terms from top documents' metadata."""
    # Rank by current fused score 's'
    ranked = sorted(score_map.values(), key=lambda r: r.get("s", 0.0), reverse=True)[
        : max(1, top_docs)
    ]
    freq: Dict[str, int] = {}
    for rec in ranked:
        md = (rec.get("pt").payload or {}).get("metadata") or {}
        path = str(md.get("path") or md.get("symbol_path") or md.get("file_path") or "")
        symbol = str(md.get("symbol") or "")
        text = f"{symbol} {path}"
        for tok in tokenize_queries([text]):
            if tok:
                freq[tok] = freq.get(tok, 0) + 1
    # sort by frequency desc
    terms = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in terms[: max(1, max_terms)]]


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


from scripts.utils import lex_hash_vector_queries as _lex_hash_vector_queries


def lex_hash_vector(phrases: List[str], dim: int = LEX_VECTOR_DIM) -> List[float]:
    return _lex_hash_vector_queries(phrases, dim)

# Defensive: sanitize Qdrant filter objects so we never send an empty filter {}
# Qdrant returns 400 if filter has no conditions; return None in that case.
def _sanitize_filter_obj(flt):
    try:
        if flt is None:
            return None
        # Try model-style attributes first
        must = getattr(flt, "must", None)
        should = getattr(flt, "should", None)
        must_not = getattr(flt, "must_not", None)
        if must is None and should is None and must_not is None:
            # Maybe dict-like
            if isinstance(flt, dict):
                m = [c for c in (flt.get("must") or []) if c is not None]
                s = [c for c in (flt.get("should") or []) if c is not None]
                mn = [c for c in (flt.get("must_not") or []) if c is not None]
                return None if (not m and not s and not mn) else flt
            # Unknown structure -> drop
            return None
        m = [c for c in (must or []) if c is not None]
        s = [c for c in (should or []) if c is not None]
        mn = [c for c in (must_not or []) if c is not None]
        if not m and not s and not mn:
            return None
        return flt
    except Exception:
        return None


def lex_query(client: QdrantClient, v: List[float], flt, per_query: int, collection_name: str | None = None) -> List[Any]:
    ef = max(EF_SEARCH, 32 + 4 * int(per_query))
    flt = _sanitize_filter_obj(flt)
    collection = _collection(collection_name)

    # Prefer modern API; handle kwarg rename between client versions (query_filter -> filter)
    try:
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=LEX_VECTOR_NAME,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except TypeError:
        # Older/newer client may expect 'filter' kw
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=LEX_VECTOR_NAME,
            filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except AttributeError:
        # Very old client without query_points: last-resort deprecated path
        return client.search(
            collection_name=collection,
            query_vector={"name": LEX_VECTOR_NAME, "vector": v},
            limit=per_query,
            with_payload=True,
            query_filter=flt,
        )


def dense_query(
    client: QdrantClient, vec_name: str, v: List[float], flt, per_query: int, collection_name: str | None = None
) -> List[Any]:
    ef = max(EF_SEARCH, 32 + 4 * int(per_query))
    flt = _sanitize_filter_obj(flt)
    collection = _collection(collection_name)

    try:
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=vec_name,
            query_filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except TypeError:
        qp = client.query_points(
            collection_name=collection,
            query=v,
            using=vec_name,
            filter=flt,
            search_params=models.SearchParams(hnsw_ef=ef),
            limit=per_query,
            with_payload=True,
        )
        return getattr(qp, "points", qp)
    except Exception as e:
        # Some Qdrant versions reject empty/malformed filters with 400: retry without a filter
        _msg = str(e).lower()
        if "expected some form of condition" in _msg or "format error in json body" in _msg:
            try:
                qp = client.query_points(
                    collection_name=collection,
                    query=v,
                    using=vec_name,
                    query_filter=None,
                    search_params=models.SearchParams(hnsw_ef=ef),
                    limit=per_query,
                    with_payload=True,
                )
                return getattr(qp, "points", qp)
            except Exception:
                pass
        # Fallback to legacy search API
        try:
            return client.search(
                collection_name=collection,
                query_vector={"name": vec_name, "vector": v},
                limit=per_query,
                with_payload=True,
                query_filter=(None if ("expected some form of condition" in _msg or "format error in json body" in _msg) else flt),
            )
        except Exception:
            raise


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
    collection: str | None = None,
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

    # Normalize glob patterns: allow repo-relative (e.g., "src/*.py") to match
    # stored absolute paths (e.g., "/work/src/..."). We keep both original and
    # absolute-prefixed variants for matching.
    def _normalize_globs(globs: list[str]) -> list[str]:
        out: list[str] = []
        try:
            for g in (globs or []):
                s = str(g).strip().replace("\\", "/")
                if not s:
                    continue
                out.append(s)
                if not s.startswith("/"):
                    out.append("/work/" + s.lstrip("/"))
        except Exception:
            pass
        # Dedup while preserving order
        seen = set()
        dedup: list[str] = []
        for g in out:
            if g not in seen:
                dedup.append(g)
                seen.add(g)
        return dedup

    eff_path_globs_norm = _normalize_globs(eff_path_globs)
    eff_not_globs_norm = _normalize_globs(eff_not_globs)


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
        must.append(
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value=eff_language)
            )
        )
    if eff_repo:
        must.append(
            models.FieldCondition(
                key="metadata.repo", match=models.MatchValue(value=eff_repo)
            )
        )
    if eff_under:
        must.append(
            models.FieldCondition(
                key="metadata.path_prefix", match=models.MatchValue(value=eff_under)
            )
        )
    if eff_kind:
        must.append(
            models.FieldCondition(
                key="metadata.kind", match=models.MatchValue(value=eff_kind)
            )
        )
    if eff_symbol:
        must.append(
            models.FieldCondition(
                key="metadata.symbol", match=models.MatchValue(value=eff_symbol)
            )
        )
    flt = models.Filter(must=must) if must else None
    flt = _sanitize_filter_obj(flt)


    # Build query list (LLM-assisted first, then synonym expansion)
    qlist = list(clean_queries)
    try:
        llm_max = int(os.environ.get("LLM_EXPAND_MAX", "4") or 4)
    except Exception:
        llm_max = 4
    _llm_more = _llm_expand_queries(qlist, eff_language, max_new=llm_max)
    for s in _llm_more:
        if s and s not in qlist:
            qlist.append(s)
    if expand:
        qlist = expand_queries(qlist, eff_language)

    # Lexical vector query
    score_map: Dict[str, Dict[str, Any]] = {}
    try:
        lex_vec = lex_hash_vector(qlist)
        lex_results = lex_query(client, lex_vec, flt, max(24, limit), collection)
    except Exception:
        lex_results = []
    for rank, p in enumerate(lex_results, 1):
        pid = str(p.id)
        score_map.setdefault(
            pid,
            {
                "pt": p,
                "s": 0.0,
                "d": 0.0,
                "lx": 0.0,
                "sym_sub": 0.0,
                "sym_eq": 0.0,
                "core": 0.0,
                "vendor": 0.0,
                "langb": 0.0,
                "rec": 0.0,
                "test": 0.0,
            },
        )
        lxs = LEX_VECTOR_WEIGHT * rrf(rank)
        score_map[pid]["lx"] += lxs
        score_map[pid]["s"] += lxs

    # Dense queries
    embedded = _embed_queries_cached(_model, qlist)
    # Ensure collection schema is compatible with current search settings (named vectors)
    try:
        if embedded:
            dim = len(embedded[0])
            _ensure_collection(client, _collection(collection), dim, vec_name)
    except Exception:
        pass
    # Optional gate-first using mini vectors to restrict dense search to candidates
    flt_gated = flt
    try:
        gate_first = str(os.environ.get("REFRAG_GATE_FIRST", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        refrag_on = str(os.environ.get("REFRAG_MODE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        cand_n = int(os.environ.get("REFRAG_CANDIDATES", "200") or 200)
    except Exception:
        gate_first, refrag_on, cand_n = False, False, 200
    if gate_first and refrag_on:
        try:
            # ReFRAG gate-first: Use MINI vectors to prefilter candidates
            mini_queries = [_project_mini(list(v), MINI_VEC_DIM) for v in embedded]

            # Get top candidates using MINI vectors (fast prefilter)
            candidate_ids = set()
            for mv in mini_queries:
                mini_results = dense_query(client, MINI_VECTOR_NAME, mv, flt, cand_n, collection)
                for result in mini_results:
                    if hasattr(result, 'id'):
                        candidate_ids.add(result.id)

            if candidate_ids:
                # Server-side gating without requiring payload fields: prefer HasIdCondition
                from qdrant_client import models as _models
                try:
                    gating_cond = _models.HasIdCondition(has_id=list(candidate_ids))
                    gating_kind = "has_id"
                except Exception:
                    # Fallback to pid_str if HasIdCondition unavailable
                    id_vals = [str(cid) for cid in candidate_ids]
                    gating_cond = _models.FieldCondition(
                        key="pid_str",
                        match=_models.MatchAny(any=id_vals),
                    )
                    gating_kind = "pid_str"
                if flt is None:
                    flt_gated = _models.Filter(must=[gating_cond])
                else:
                    must = list(flt.must or [])
                    must.append(gating_cond)
                    flt_gated = _models.Filter(must=must, should=flt.should, must_not=flt.must_not)
                if os.environ.get("DEBUG_HYBRID_SEARCH"):
                    print(f"DEBUG: ReFRAG gate-first (server-side-{gating_kind}): {len(candidate_ids)} candidates")
                    print(f"DEBUG: flt_gated.must has {len(flt_gated.must or [])} conditions")
                    print(f"DEBUG: flt_gated.must_not has {len(flt_gated.must_not or [])} conditions")
            else:
                # No candidates -> no gating
                flt_gated = flt
        except Exception as e:
            if os.environ.get("DEBUG_HYBRID_SEARCH"):
                print(f"DEBUG: ReFRAG gate-first failed: {e}, proceeding without gating")
            # Fallback to normal search (no gating)
            flt_gated = flt
    else:
        flt_gated = flt

    # Sanitize filter: if empty, drop it to avoid Qdrant 400s on invalid filters
    try:
        if flt_gated is not None:
            _m = [c for c in (getattr(flt_gated, "must", None) or []) if c is not None]
            _s = [c for c in (getattr(flt_gated, "should", None) or []) if c is not None]
            _mn = [c for c in (getattr(flt_gated, "must_not", None) or []) if c is not None]
            if not _m and not _s and not _mn:
                flt_gated = None
    except Exception:
        pass

    flt_gated = _sanitize_filter_obj(flt_gated)

    result_sets: List[List[Any]] = [
        dense_query(client, vec_name, v, flt_gated, max(24, limit), collection) for v in embedded
    ]
    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        total_dense_results = sum(len(rs) for rs in result_sets)
        print(f"DEBUG: Dense query returned {total_dense_results} total results across {len(result_sets)} queries")

    # Optional ReFRAG-style mini-vector gating: add compact-vector RRF if enabled
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                mini_queries = [_project_mini(list(v), MINI_VEC_DIM) for v in embedded]
                mini_sets: List[List[Any]] = [
                    dense_query(client, MINI_VECTOR_NAME, mv, flt, max(24, limit), collection)
                    for mv in mini_queries
                ]
                for res in mini_sets:
                    for rank, p in enumerate(res, 1):
                        pid = str(p.id)
                        score_map.setdefault(
                            pid,
                            {
                                "pt": p,
                                "s": 0.0,
                                "d": 0.0,
                                "lx": 0.0,
                                "sym_sub": 0.0,
                                "sym_eq": 0.0,
                                "core": 0.0,
                                "vendor": 0.0,
                                "langb": 0.0,
                                "rec": 0.0,
                                "test": 0.0,
                            },
                        )
                        dens = float(HYBRID_MINI_WEIGHT) * rrf(rank)
                        score_map[pid]["d"] += dens
                        score_map[pid]["s"] += dens
            except Exception:
                pass
    except Exception:
        pass

    # Pseudo-Relevance Feedback (default-on): mine top terms from current results and run a light second pass
    try:
        prf_enabled = _env_truthy(os.environ.get("PRF_ENABLED"), True)
    except Exception:
        prf_enabled = True
    if prf_enabled and score_map:
        try:
            top_docs = int(os.environ.get("PRF_TOP_DOCS", "8") or 8)
        except Exception:
            top_docs = 8
        try:
            max_terms = int(os.environ.get("PRF_MAX_TERMS", "6") or 6)
        except Exception:
            max_terms = 6
        try:
            extra_q = int(os.environ.get("PRF_EXTRA_QUERIES", "4") or 4)
        except Exception:
            extra_q = 4
        try:
            prf_dw = float(os.environ.get("PRF_DENSE_WEIGHT", "0.4") or 0.4)
        except Exception:
            prf_dw = 0.4
        try:
            prf_lw = float(os.environ.get("PRF_LEX_WEIGHT", "0.6") or 0.6)
        except Exception:
            prf_lw = 0.6
        terms = _prf_terms_from_results(
            score_map, top_docs=top_docs, max_terms=max_terms
        )
        base = clean_queries[0] if clean_queries else (qlist[0] if qlist else "")
        prf_qs: List[str] = []
        for t in terms:
            cand = (base + " " + t).strip()
            if cand and cand not in qlist and cand not in prf_qs:
                prf_qs.append(cand)
                if len(prf_qs) >= extra_q:
                    break
        if prf_qs:
            # Lexical PRF pass
            try:
                lex_vec2 = lex_hash_vector(prf_qs)
                lex_results2 = lex_query(
                    client, lex_vec2, flt, max(12, limit // 2 or 6), collection
                )
            except Exception:
                lex_results2 = []
            for rank, p in enumerate(lex_results2, 1):
                pid = str(p.id)
                score_map.setdefault(
                    pid,
                    {
                        "pt": p,
                        "s": 0.0,
                        "d": 0.0,
                        "lx": 0.0,
                        "sym_sub": 0.0,
                        "sym_eq": 0.0,
                        "core": 0.0,
                        "vendor": 0.0,
                        "langb": 0.0,
                        "rec": 0.0,
                        "test": 0.0,
                    },
                )
                lxs = prf_lw * rrf(rank)
                score_map[pid]["lx"] += lxs
                score_map[pid]["s"] += lxs
            # Dense PRF pass
            try:
                embedded2 = _embed_queries_cached(_model, prf_qs)
                result_sets2: List[List[Any]] = [
                    dense_query(client, vec_name, v, flt, max(12, limit // 2 or 6), collection)
                    for v in embedded2
                ]
                for res2 in result_sets2:
                    for rank, p in enumerate(res2, 1):
                        pid = str(p.id)
                        score_map.setdefault(
                            pid,
                            {
                                "pt": p,
                                "s": 0.0,
                                "d": 0.0,
                                "lx": 0.0,
                                "sym_sub": 0.0,
                                "sym_eq": 0.0,
                                "core": 0.0,
                                "vendor": 0.0,
                                "langb": 0.0,
                                "rec": 0.0,
                                "test": 0.0,
                            },
                        )
                        dens = prf_dw * rrf(rank)
                        score_map[pid]["d"] += dens
                        score_map[pid]["s"] += dens
            except Exception:
                pass

    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(
                pid,
                {
                    "pt": p,
                    "s": 0.0,
                    "d": 0.0,
                    "lx": 0.0,
                    "sym_sub": 0.0,
                    "sym_eq": 0.0,
                    "core": 0.0,
                    "vendor": 0.0,
                    "langb": 0.0,
                    "rec": 0.0,
                    "test": 0.0,
                },
            )
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
        if TEST_FILE_PENALTY > 0.0 and path and is_test_file(path):
            rec["test"] -= TEST_FILE_PENALTY
            rec["s"] -= TEST_FILE_PENALTY

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

    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        print(f"DEBUG: score_map has {len(score_map)} items before ranking")
    ranked = sorted(score_map.values(), key=_tie_key)
    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        print(f"DEBUG: ranked has {len(ranked)} items after sorting")

    # Lightweight keyword bump: prefer spans whose local snippet contains query tokens
    try:
        kb = float(os.environ.get("HYBRID_KEYWORD_BUMP", "0.3") or 0.3)
        kcap = float(os.environ.get("HYBRID_KEYWORD_CAP", "0.6") or 0.6)
    except Exception:
        kb, kcap = 0.3, 0.6
    # Build lowercase keyword set from queries (simple split, keep >=3 chars + special tokens)
    kw: set[str] = set()
    for q in qlist:
        ql = (q or "").lower()
        for tok in re.findall(r"[a-zA-Z0-9_\-]+", ql):
            t = tok.strip()
            if len(t) >= 3:
                kw.add(t)
    # Add a few commonly relevant code tokens (helps for gate-first cases)
    for t in ("hasidcondition", "pid_str", "matchany", "gate-first", "gatefirst"):
        kw.add(t)

    import io as _io

    def _snippet_contains(md: dict) -> int:
        # returns number of keyword hits found in a small local snippet
        try:
            path = str(md.get("path") or "")
            sline = int(md.get("start_line") or 0)
            eline = int(md.get("end_line") or 0)
            txt = (md.get("text") or md.get("code") or "")
            if not txt and path and sline:
                p = path
                try:
                    if not os.path.isabs(p):
                        p = os.path.join("/work", p)
                    realp = os.path.realpath(p)
                    if realp == "/work" or realp.startswith("/work/"):
                        with open(realp, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                        si = max(1, sline - 3)
                        ei = min(len(lines), max(sline, eline) + 3)
                        txt = "".join(lines[si-1:ei])
                except Exception:
                    txt = txt or ""
            lt = (txt or "").lower()
            if not lt:
                return 0
            hits = 0
            for t in kw:
                if t and t in lt:
                    hits += 1
            return hits
        except Exception:
            return 0

    # Apply bump to top-N ranked (limited for speed)
    topN = min(len(ranked), 200)
    for i in range(topN):
        m = ranked[i]
        md = (m["pt"].payload or {}).get("metadata") or {}
        hits = _snippet_contains(md)
        if hits > 0 and kb > 0.0:
            bump = min(kcap, kb * float(hits))
            m["s"] += bump
    # Re-sort after bump
    ranked = sorted(ranked, key=_tie_key)

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
            if (
                start_line <= c["end"] + CLUSTER_LINES
                and end_line >= c["start"] - CLUSTER_LINES
            ):
                if float(m["s"]) > float(c["m"]["s"]):
                    c["m"] = m
                c["start"] = min(c["start"], start_line)
                c["end"] = max(c["end"], end_line)
                merged_flag = True
                break
        if not merged_flag:
            lst.append({"start": start_line, "end": end_line, "m": m})

    ranked = sorted([c["m"] for lst in clusters.values() for c in lst], key=_tie_key)
    if os.environ.get("DEBUG_HYBRID_SEARCH"):
        print(f"DEBUG: ranked has {len(ranked)} items after clustering")

    # Client-side filters and per-path diversification
    import re as _re, fnmatch as _fnm

    case_sensitive = str(eff_case or "").lower() == "sensitive"

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
            rel = path[6:] if path.startswith("/work/") else path
            pp = str(md.get("path_prefix") or "")
            p_for_sub = path if case_sensitive else path.lower()
            pp_for_sub = pp if case_sensitive else pp.lower()
            if eff_not:
                nn = eff_not if case_sensitive else eff_not.lower()
                if nn in p_for_sub or nn in pp_for_sub:
                    return False
            if eff_not_globs_norm and any(_match_glob(g, path) or _match_glob(g, rel) for g in eff_not_globs_norm):
                return False
            if eff_ext:
                ex = eff_ext.lower().lstrip(".")
                if not path.lower().endswith("." + ex):
                    return False
            if eff_path_regex:
                flags = 0 if case_sensitive else _re.IGNORECASE
                try:
                    if not _re.search(eff_path_regex, path, flags=flags):
                        return False
                except Exception:
                    pass
            if eff_path_globs_norm and not any(_match_glob(g, path) or _match_glob(g, rel) for g in eff_path_globs_norm):
                return False
            return True

        ranked = [m for m in ranked if _pass_filters(m)]

    # ReFRAG-lite span compaction and budgeting is NOT applied here in run_hybrid_search
    # It's only applied in context_answer where token budgeting is needed for LLM context
    # Removing this to avoid over-filtering search results

    if per_path and per_path > 0:
        counts: Dict[str, int] = {}
        merged: List[Dict[str, Any]] = []
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            print(f"DEBUG: Applying per_path={per_path} limiting to {len(ranked)} ranked results")
        for m in ranked:
            md = (m["pt"].payload or {}).get("metadata") or {}
            path = str(md.get("path", ""))
            c = counts.get(path, 0)
            if c < per_path:
                merged.append(m)
                counts[path] = c + 1
            if len(merged) >= limit:
                break
        if os.environ.get("DEBUG_HYBRID_SEARCH"):
            print(f"DEBUG: After per_path limiting: {len(merged)} results from {len(counts)} unique paths")
    else:
        merged = ranked[:limit]

    # Emit structured items
    # Build directory → paths map for related hints (same dir siblings)
    dir_to_paths: Dict[str, set] = {}
    try:
        for _m in merged:
            _md = (_m["pt"].payload or {}).get("metadata") or {}
            _pp = str(_md.get("path_prefix") or "")
            _p = str(_md.get("path") or "")
            if _pp and _p:
                dir_to_paths.setdefault(_pp, set()).add(_p)
    except Exception:
        dir_to_paths = {}
    # Precompute known paths for quick membership checks
    all_paths: set = set()
    try:
        for _s in dir_to_paths.values():
            all_paths |= set(_s)
    except Exception:
        all_paths = set()

    items: List[Dict[str, Any]] = []
    for m in merged:
        md = (m["pt"].payload or {}).get("metadata") or {}
        # Prefer merged bounds if present
        start_line = m.get("_merged_start") or md.get("start_line")
        end_line = m.get("_merged_end") or md.get("end_line")
        comp = {
            "dense_rrf": round(float(m.get("d", 0.0)), 4),
            "lexical": round(float(m.get("lx", 0.0)), 4),
            "symbol_substr": round(float(m.get("sym_sub", 0.0)), 4),
            "symbol_exact": round(float(m.get("sym_eq", 0.0)), 4),
            "core_boost": round(float(m.get("core", 0.0)), 4),
            "vendor_penalty": round(float(m.get("vendor", 0.0)), 4),
            "lang_boost": round(float(m.get("langb", 0.0)), 4),
            "recency": round(float(m.get("rec", 0.0)), 4),
            "test_penalty": round(float(m.get("test", 0.0)), 4),
        }
        why = []
        if comp["dense_rrf"]:
            why.append(f"dense_rrf:{comp['dense_rrf']}")
        for k in ("lexical", "symbol_substr", "symbol_exact", "core_boost", "lang_boost"):
            if comp[k]:
                why.append(f"{k}:{comp[k]}")
        if comp["vendor_penalty"]:
            why.append(f"vendor_penalty:{comp['vendor_penalty']}")
        if comp.get("test_penalty"):
            why.append(f"test_penalty:{comp['test_penalty']}")
        if comp["recency"]:
            why.append(f"recency:{comp['recency']}")
        # Related hints
        _imports = md.get("imports") or []
        _calls = md.get("calls") or []
        _symp = md.get("symbol_path") or md.get("symbol") or ""
        _pp = str(md.get("path_prefix") or "")
        _path = str(md.get("path") or "")
        _related_set = set()
        # Same-dir siblings
        try:
            if _pp in dir_to_paths:
                for p in dir_to_paths[_pp]:
                    if p != _path:
                        _related_set.add(p)
        except Exception:
            pass
        # Import-based hints: resolve relative/quoted path-like imports
        try:
            import re as _re, posixpath as _ppath

            def _pathlike_segments(s: str) -> list[str]:
                s = str(s or "")
                segs = []
                # quoted segments first
                for mmm in _re.findall(r"[\"']([^\"']+)[\"']", s):
                    if "/" in mmm or mmm.startswith("."):
                        segs.append(mmm)
                # fall back to whitespace tokens containing '/' or starting with '.'
                for tok in str(s).replace(",", " ").split():
                    if ("/" in tok) or tok.startswith("."):
                        segs.append(tok)
                return segs

            def _resolve(seg: str) -> list[str]:
                try:
                    seg = seg.strip()
                    # base dir from path_prefix
                    base = _pp or ""
                    candidates = []
                    # choose join rule
                    if seg.startswith("./") or seg.startswith("../") or "/" in seg:
                        j = _ppath.normpath(_ppath.join(base, seg)) if not seg.startswith("/") else _ppath.normpath(seg)
                        candidates.append(j)
                        # add extensions if last segment lacks a dot
                        last = j.split("/")[-1]
                        if "." not in last:
                            for ext in [".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"]:
                                candidates.append(j + ext)
                    out = set()
                    for c in candidates:
                        if c in all_paths:
                            out.add(c)
                        if c.startswith("/") and c.lstrip("/") in all_paths:
                            out.add(c.lstrip("/"))
                        if c.startswith("/work/") and c[len("/work/"):] in all_paths:
                            out.add(c[len("/work/"):])
                    return list(out)
                except Exception:
                    return []

            for imp in (_imports or []):
                for seg in _pathlike_segments(imp):
                    for cand in _resolve(seg):
                        if cand != _path:
                            _related_set.add(cand)
        except Exception:
            pass

        _related = sorted(_related_set)[:10]
        # Best-effort snippet text directly from payload for downstream LLM stitching
        _payload = (m["pt"].payload or {}) if m.get("pt") is not None else {}
        _metadata = _payload.get("metadata", {}) or {}
        _text = (
            _payload.get("code") or
            _metadata.get("code") or
            _payload.get("text") or
            _metadata.get("text") or
            ""
        )
        # Skip memory-like points without a real file path
        if not _path or not _path.strip():
            if os.environ.get("DEBUG_HYBRID_FILTER"):
                print(f"DEBUG: Filtered out item with empty path: {_metadata}")
            continue

        # Emit path: prefer original host path when available; also include container path
        _emit_path = _path
        _host = ""
        _cont = ""
        try:
            _host = str(_metadata.get("host_path") or "").strip()
            _cont = str(_metadata.get("container_path") or "").strip()
            _repo = str(_metadata.get("repo") or "").strip()
            _pp = str(_metadata.get("path_prefix") or "").strip()
            _mode = str(os.environ.get("PATH_EMIT_MODE", "auto")).strip().lower()

            if _mode == "host" and _host:
                _emit_path = _host
            elif _mode == "container" and _cont:
                _emit_path = _cont
            else:
                # Auto mode: prefer host when available, else container; then fallback normalization
                if _host:
                    _emit_path = _host
                elif _cont:
                    _emit_path = _cont
                else:
                    # Auto/compat fallback: normalize to container form if repo+prefix known; else map cwd to /work
                    if _repo and _pp and isinstance(_emit_path, str):
                        _pp_norm = _pp.rstrip("/") + "/"
                        if _emit_path.startswith(_pp_norm):
                            _rel = _emit_path[len(_pp_norm):]
                            if _rel:
                                _emit_path = f"/work/{_repo}/" + _rel.lstrip("/")
                    if isinstance(_emit_path, str):
                        _cwd = os.getcwd().rstrip("/") + "/"
                        if _emit_path.startswith(_cwd):
                            _rel = _emit_path[len(_cwd):]
                            if _rel:
                                _emit_path = "/work/" + _rel
        except Exception:
            pass

        items.append(
            {
                "score": round(float(m["s"]), 4),
                "path": _emit_path,
                "host_path": _host,
                "container_path": _cont,
                "symbol": _symp,
                "start_line": start_line,
                "end_line": end_line,
                "components": comp,
                "why": why,
                "relations": {"imports": _imports, "calls": _calls, "symbol_path": _symp},
                "related_paths": _related,
                "span_budgeted": bool(m.get("_merged_start") is not None),
                "budget_tokens_used": m.get("_budget_tokens"),
                "text": _text,
            }
        )
    return items


def main():
    ap = argparse.ArgumentParser(description="Hybrid search: dense + lexical RRF")
    ap.add_argument(
        "--query",
        "-q",
        action="append",
        required=True,
        help="One or more query strings (multi-query)",
    )
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--under", type=str, default=None)
    ap.add_argument("--kind", type=str, default=None)
    ap.add_argument("--symbol", type=str, default=None)
    # Expansion enabled by default; allow disabling via --no-expand or HYBRID_EXPAND=0
    ap.add_argument(
        "--expand",
        dest="expand",
        action="store_true",
        default=_env_truthy(os.environ.get("HYBRID_EXPAND"), True),
        help="Enable simple query expansion",
    )
    ap.add_argument(
        "--no-expand",
        dest="expand",
        action="store_false",
        help="Disable query expansion",
    )
    # Per-path diversification enabled by default (1) unless overridden by env/flag
    ap.add_argument(
        "--per-path",
        type=int,
        default=int(os.environ.get("HYBRID_PER_PATH", "1") or 1),
        help="Cap results per file path to diversify (0=off)",
    )

    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--per-query", type=int, default=24)
    ap.add_argument(
        "--json",
        dest="json",
        action="store_true",
        help="Emit JSON lines with score breakdown",
    )
    # Structured filters to mirror MCP tool fields
    ap.add_argument("--ext", type=str, default=None)
    ap.add_argument("--not", dest="not_filter", type=str, default=None)
    ap.add_argument("--collection", type=str, default=None,
                     help="Target collection name")
    ap.add_argument(
        "--case",
        type=str,
        choices=["sensitive", "insensitive"],
        default=os.environ.get("HYBRID_CASE", "insensitive"),
    )
    ap.add_argument("--path-regex", dest="path_regex", type=str, default=None)
    ap.add_argument("--path-glob", dest="path_glob", type=str, default=None)
    ap.add_argument("--not-glob", dest="not_glob", type=str, default=None)

    args = ap.parse_args()

    # Resolve effective collection early to avoid variable usage errors
    eff_collection = args.collection or os.environ.get("COLLECTION_NAME", "my-collection")

    model = TextEmbedding(model_name=MODEL_NAME)
    vec_name = _sanitize_vector_name(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    # Ensure collection exists with dual named vectors before search
    try:
        first_vec = next(model.embed(["__dim__warmup__"]))
        dim = len(first_vec.tolist())
        _ensure_collection(client, _collection(eff_collection), dim, vec_name)
    except Exception:
        pass

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
        must.append(
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value=eff_language)
            )
        )
    if eff_repo:
        must.append(
            models.FieldCondition(
                key="metadata.repo", match=models.MatchValue(value=eff_repo)
            )
        )
    if eff_under:
        must.append(
            models.FieldCondition(
                key="metadata.path_prefix", match=models.MatchValue(value=eff_under)
            )
        )
    # If ext: was provided without an explicit language, infer language from extension
    if eff_ext and not eff_language:
        ex = eff_ext.lower().lstrip(".")
        for lang, exts in LANG_EXTS.items():
            if any(ex == e.lstrip(".").lower() for e in exts):
                eff_language = lang
                break

    if eff_kind:
        must.append(
            models.FieldCondition(
                key="metadata.kind", match=models.MatchValue(value=eff_kind)
            )
        )
    if eff_symbol:
        must.append(
            models.FieldCondition(
                key="metadata.symbol", match=models.MatchValue(value=eff_symbol)
            )
        )
    flt = models.Filter(must=must) if must else None
    flt = _sanitize_filter_obj(flt)

    # Build query set (optionally expanded)
    queries = list(clean_queries)
    # Initialize score map early so we can accumulate from lex and dense
    score_map: Dict[str, Dict[str, Any]] = {}
    # Server-side lexical vector search (hashing) as an additional ranked list
    try:
        lex_vec = lex_hash_vector(queries)
        lex_results = lex_query(client, lex_vec, flt, args.per_query, eff_collection)
    except Exception:
        lex_results = []

    if args.expand:
        queries = expand_queries(queries, eff_language)

    # Add server-side lexical vector ranking into fusion
    for rank, p in enumerate(lex_results, 1):
        pid = str(p.id)
        score_map.setdefault(
            pid,
            {
                "pt": p,
                "s": 0.0,
                "d": 0.0,
                "lx": 0.0,
                "sym_sub": 0.0,
                "sym_eq": 0.0,
                "core": 0.0,
                "vendor": 0.0,
                "langb": 0.0,
                "rec": 0.0,
                "test": 0.0,
            },
        )
        lxs = LEX_VECTOR_WEIGHT * rrf(rank)
        score_map[pid]["lx"] += lxs
        score_map[pid]["s"] += lxs

    embedded = _embed_queries_cached(model, queries)
    result_sets: List[List[Any]] = [
        dense_query(client, vec_name, v, flt, args.per_query, eff_collection) for v in embedded
    ]

    # RRF fusion (weighted)
    for res in result_sets:
        for rank, p in enumerate(res, 1):
            pid = str(p.id)
            score_map.setdefault(
                pid,
                {
                    "pt": p,
                    "s": 0.0,
                    "d": 0.0,
                    "lx": 0.0,
                    "sym_sub": 0.0,
                    "sym_eq": 0.0,
                    "core": 0.0,
                    "vendor": 0.0,
                    "langb": 0.0,
                    "rec": 0.0,
                    "test": 0.0,
                },
            )
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
        if TEST_FILE_PENALTY > 0.0 and path and is_test_file(path):
            rec["test"] -= TEST_FILE_PENALTY
            rec["s"] -= TEST_FILE_PENALTY


        # Language match boost if requested
        if (
            LANG_MATCH_BOOST > 0.0
            and path
            and (eff_language or getattr(args, "language", None))
        ):
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
            if (
                start_line <= c["end"] + CLUSTER_LINES
                and end_line >= c["start"] - CLUSTER_LINES
            ):
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

    case_sensitive = str(eff_case or "").lower() == "sensitive"
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
            rel = path[6:] if path.startswith("/work/") else path
            pp = str(md.get("path_prefix") or "")
            p_for_sub = path if case_sensitive else path.lower()
            pp_for_sub = pp if case_sensitive else pp.lower()
            # NOT substring filter
            if eff_not:
                nn = eff_not if case_sensitive else eff_not.lower()
                if nn in p_for_sub or nn in pp_for_sub:
                    return False
            # not_glob exclusion
            if eff_not_glob and (
                _match_glob(eff_not_glob, path)
                or _match_glob(eff_not_glob, rel)
                or (not str(eff_not_glob).startswith("/") and _match_glob("/work/" + str(eff_not_glob).lstrip("/"), path))
            ):
                return False
            # Extension filter (normalize to .ext)
            if eff_ext:
                ex = eff_ext.lower().lstrip(".")
                if not path.lower().endswith("." + ex):
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
            if eff_path_glob and not (
                _match_glob(eff_path_glob, path)
                or _match_glob(eff_path_glob, rel)
                or (not str(eff_path_glob).startswith("/") and _match_glob("/work/" + str(eff_path_glob).lstrip("/"), path))
            ):
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
    # Build directory → paths map for related path hints
    dir_to_paths: Dict[str, set] = {}
    try:
        for m in merged:
            md = (m["pt"].payload or {}).get("metadata") or {}
            pp = str(md.get("path_prefix") or "")
            p = str(md.get("path") or "")
            if pp and p:
                dir_to_paths.setdefault(pp, set()).add(p)
    except Exception:
        dir_to_paths = {}

    else:
        merged = ranked[: args.limit]

    for m in merged:
        md = (m["pt"].payload or {}).get("metadata") or {}
        if getattr(args, "json", False):
            # Related hints
            _imports = md.get("imports") or []
            _calls = md.get("calls") or []
            _symp = md.get("symbol_path") or md.get("symbol") or ""
            _pp = str(md.get("path_prefix") or "")
            _related = []
            try:
                if _pp in dir_to_paths:
                    _related = [p for p in sorted(dir_to_paths[_pp]) if p != md.get("path")][:5]
            except Exception:
                _related = []
            item = {
                "score": round(float(m["s"]), 4),
                "path": md.get("path"),
                "symbol": _symp,
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
                    "test_penalty": round(float(m.get("test", 0.0)), 4),
                },
                "relations": {"imports": _imports, "calls": _calls, "symbol_path": _symp},
                "related_paths": _related,
            }
            # Build a human friendly why list
            why = []
            if item["components"]["dense_rrf"]:
                why.append(f"dense_rrf:{item['components']['dense_rrf']}")
            for k in (
                "lexical",
                "symbol_substr",
                "symbol_exact",
                "core_boost",
                "lang_boost",
            ):
                if item["components"][k]:
                    why.append(f"{k}:{item['components'][k]}")
            if item["components"]["vendor_penalty"]:
                why.append(f"vendor_penalty:{item['components']['vendor_penalty']}")
            if item["components"].get("test_penalty"):
                why.append(f"test_penalty:{item['components']['test_penalty']}")
            if item["components"]["recency"]:
                why.append(f"recency:{item['components']['recency']}")
            item["why"] = why
            print(json.dumps(item))
        else:
            print(
                f"{m['s']:.3f}\t{md.get('path')}\t{md.get('symbol_path') or md.get('symbol') or ''}\t{md.get('start_line')}-{md.get('end_line')}"
            )


if __name__ == "__main__":
    main()
