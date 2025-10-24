from __future__ import annotations


# Helper: detect repository name automatically (no REPO_NAME env needed)
def _detect_repo_name_from_path(path: Path) -> str:
    try:
        import subprocess, os as _os

        base = path if path.is_dir() else path.parent
        r = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        top = r.stdout.strip()
        if r.returncode == 0 and top:
            return Path(top).name or "workspace"
    except Exception:
        pass
    # Fallback: walk up to find a .git folder
    try:
        cur = path if path.is_dir() else path.parent
        for p in [cur] + list(cur.parents):
            try:
                if (p / ".git").exists():
                    return p.name or "workspace"
            except Exception:
                continue
    except Exception:
        pass
    # Last resort: directory name
    try:
        return (path if path.is_dir() else path.parent).name or "workspace"
    except Exception:
        return "workspace"


#!/usr/bin/env python3
import os
import sys
import argparse
import hashlib
import re
import ast
import time
from pathlib import Path
from typing import List, Dict, Iterable

# Ensure project root is on sys.path when run as a script (so 'scripts' package imports work)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding



from datetime import datetime
try:
    from scripts.workspace_state import (
        update_indexing_status,
        update_last_activity,
        update_workspace_state,
    )
except Exception:
    # State integration is optional; continue if not available
    update_indexing_status = None  # type: ignore
    update_last_activity = None  # type: ignore
    update_workspace_state = None  # type: ignore

# Optional Tree-sitter import (graceful fallback)
try:
    from tree_sitter import Parser  # type: ignore
    from tree_sitter_languages import get_language  # type: ignore

    _TS_AVAILABLE = True
except Exception:  # pragma: no cover
    Parser = None  # type: ignore
    get_language = None  # type: ignore
    _TS_AVAILABLE = False


_TS_WARNED = False


def _use_tree_sitter() -> bool:
    global _TS_WARNED
    want = os.environ.get("USE_TREE_SITTER", "").lower() in {"1", "true", "yes", "on"}
    if want and not _TS_AVAILABLE and not _TS_WARNED:
        print(
            "[WARN] USE_TREE_SITTER=1 but tree-sitter libs not available; falling back to regex heuristics"
        )
        _TS_WARNED = True
    return _TS_AVAILABLE and want


CODE_EXTS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".sql": "sql",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".json": "json",
    ".tf": "terraform",
    ".csx": "csharp",
    ".cshtml": "razor",
    ".razor": "razor",
    ".csproj": "xml",
    ".config": "xml",
    ".resx": "xml",

}

# --- Named vector config ---
LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)
# Optional mini vector (ReFRAG-style gating); conditionally created by REFRAG_MODE
MINI_VECTOR_NAME = os.environ.get("MINI_VECTOR_NAME", "mini")
MINI_VEC_DIM = int(os.environ.get("MINI_VEC_DIM", "64") or 64)

# Lightweight hashing-trick sparse vector (fixed-size dense) for lexical signals
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

# Random +/-1 projection (Gaussian-sign) for compact mini vectors; cached by (in_dim,out_dim,seed)
_MINI_PROJ_CACHE: dict[tuple[int, int, int], list[list[float]]] = {}


def _get_mini_proj(
    in_dim: int, out_dim: int, seed: int | None = None
) -> list[list[float]]:
    import math, random

    s = int(os.environ.get("MINI_VEC_SEED", "1337")) if seed is None else int(seed)
    key = (in_dim, out_dim, s)
    M = _MINI_PROJ_CACHE.get(key)
    if M is None:
        rnd = random.Random(s)
        scale = 1.0 / math.sqrt(out_dim)
        # Dense Rademacher matrix (+/-1) scaled; good enough for fast gating
        M = [
            [scale * (1.0 if rnd.random() < 0.5 else -1.0) for _ in range(out_dim)]
            for _ in range(in_dim)
        ]
        _MINI_PROJ_CACHE[key] = M
    return M


def project_mini(vec: list[float], out_dim: int | None = None) -> list[float]:
    import math

    if not vec:
        return [0.0] * (int(out_dim or MINI_VEC_DIM))
    od = int(out_dim or MINI_VEC_DIM)
    M = _get_mini_proj(len(vec), od)
    out = [0.0] * od
    # y = x @ M
    for i, val in enumerate(vec):
        if val == 0.0:
            continue
        row = M[i]
        for j in range(od):
            out[j] += val * row[j]
    # L2 normalize to keep scale consistent
    norm = (sum(x * x for x in out) or 0.0) ** 0.5 or 1.0
    return [x / norm for x in out]


def _split_ident_lex(s: str):
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def _lex_hash_vector(text: str, dim: int = LEX_VECTOR_DIM) -> list[float]:
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    # Tokenize identifiers & words
    toks = _split_ident_lex(text)
    if not toks:
        return vec
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize (avoid huge magnitudes)
    import math

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _git_metadata(file_path: Path) -> tuple[int, int, int]:
    """Return (last_modified_at, churn_count, author_count) using git when available.
    Fallbacks to fs mtime and zeros when not in a repo.
    """
    try:
        import subprocess

        fp = str(file_path)
        # last commit unix timestamp (%ct)
        ts = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout.strip()
        last_ts = int(ts) if ts.isdigit() else int(file_path.stat().st_mtime)
        # churn: number of commits touching this file (bounded)
        churn_s = subprocess.run(
            ["git", "rev-list", "--count", "HEAD", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout.strip()
        churn = int(churn_s) if churn_s.isdigit() else 0
        # author count
        authors = subprocess.run(
            ["git", "shortlog", "-s", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout
        author_count = len([ln for ln in authors.splitlines() if ln.strip()])
        return last_ts, churn, author_count
    except Exception:
        try:
            return int(file_path.stat().st_mtime), 0, 0
        except Exception:
            return int(time.time()), 0, 0


# --- Exclusions (configurable) ---
# Defaults can be overridden by .qdrantignore, env, or CLI
_DEFAULT_EXCLUDE_DIRS = [
    "/models",
    "/.vs",

    "/node_modules",
    "/dist",
    "/build",
    "/.venv",
    "/venv",
    "/__pycache__",
    "bin",
    "obj",
    "TestResults",

    "/.git",
]
_DEFAULT_EXCLUDE_FILES = [
    "*.onnx",
    "*.bin",
    "*.safetensors",
    "tokenizer.json",
    "*.whl",
    "*.tar.gz",
]


def _env_truthy(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def hash_id(text: str, path: str, start: int, end: int) -> int:
    h = hashlib.sha1(
        f"{path}:{start}-{end}\n{text}".encode("utf-8", errors="ignore")
    ).hexdigest()
    return int(h[:16], 16)


class _Excluder:
    def __init__(self, root: Path):
        self.root = root
        self.dir_prefixes = []  # absolute like /path/sub
        self.file_globs = []  # fnmatch patterns

        # Defaults
        use_defaults = _env_truthy(os.environ.get("QDRANT_DEFAULT_EXCLUDES"), True)
        if use_defaults:
            self.dir_prefixes.extend(_DEFAULT_EXCLUDE_DIRS)
            self.file_globs.extend(_DEFAULT_EXCLUDE_FILES)

        # .qdrantignore
        ignore_file = os.environ.get("QDRANT_IGNORE_FILE", ".qdrantignore")
        ig_path = root / ignore_file
        if ig_path.exists():
            for raw in ig_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                self._add_pattern(line)

        # Extra excludes via env (comma separated)
        extra = os.environ.get("QDRANT_EXCLUDES", "").strip()
        if extra:
            for pat in [p.strip() for p in extra.split(",") if p.strip()]:
                self._add_pattern(pat)

    def _add_pattern(self, pat: str):
        # Normalize to leading-slash for prefixes
        has_wild = any(ch in pat for ch in "*?[")
        if pat.startswith("/") and not has_wild:
            # Treat as directory prefix if no wildcard
            self.dir_prefixes.append(pat.rstrip("/"))
        else:
            # Treat as file glob (match against relpath and basename)
            self.file_globs.append(pat.lstrip("/"))

    def exclude_dir(self, rel: str) -> bool:
        # rel like /a/b
        for pref in self.dir_prefixes:
            if rel == pref or rel.startswith(pref + "/"):
                return True
        # Also allow dir name-only patterns in file_globs (e.g., node_modules)
        base = rel.rsplit("/", 1)[-1]
        for g in self.file_globs:
            # Match bare dir names without wildcards
            if g and all(ch not in g for ch in "*?[") and base == g:
                return True
        return False

    def exclude_file(self, rel: str) -> bool:
        import fnmatch

        # Try matching whole rel path and basename
        base = rel.rsplit("/", 1)[-1]
        for g in self.file_globs:
            if fnmatch.fnmatch(rel.lstrip("/"), g) or fnmatch.fnmatch(base, g):
                return True
        return False


def iter_files(root: Path) -> Iterable[Path]:
    # Allow passing a single file
    if root.is_file():
        if root.suffix.lower() in CODE_EXTS:
            yield root
        return

    excl = _Excluder(root)
    # Use os.walk to prune directories for performance
    for dirpath, dirnames, filenames in os.walk(root):
        # Compute rel path like /a/b from root
        rel_dir = "/" + str(
            Path(dirpath).resolve().relative_to(root.resolve())
        ).replace(os.sep, "/")
        if rel_dir == "/.":
            rel_dir = "/"
        # Prune excluded directories in-place
        keep = []
        for d in dirnames:
            rel = (rel_dir.rstrip("/") + "/" + d).replace("//", "/")
            if excl.exclude_dir(rel):
                continue
            keep.append(d)
        dirnames[:] = keep

        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix.lower() not in CODE_EXTS:
                continue
            relf = (rel_dir.rstrip("/") + "/" + f).replace("//", "/")
            if excl.exclude_file(relf):
                continue
            yield p


def chunk_lines(text: str, max_lines: int = 120, overlap: int = 20) -> List[Dict]:
    lines = text.splitlines()
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        j = min(n, i + max_lines)
        chunk = "\n".join(lines[i:j])
        chunks.append({"text": chunk, "start": i + 1, "end": j})
        if j == n:
            break
        i = max(j - overlap, i + 1)
    return chunks


def chunk_semantic(
    text: str, language: str, max_lines: int = 120, overlap: int = 20
) -> List[Dict]:
    """AST-aware chunking that tries to keep complete functions/classes together."""
    if not _use_tree_sitter() or language not in ("python", "javascript", "typescript"):
        # Fallback to line-based chunking
        return chunk_lines(text, max_lines, overlap)

    lines = text.splitlines()
    n = len(lines)

    # Extract symbols with line ranges
    symbols = _extract_symbols(language, text)
    if not symbols:
        return chunk_lines(text, max_lines, overlap)

    # Sort symbols by start line
    symbols.sort(key=lambda s: s.start)

    chunks = []
    i = 0  # Current line index (0-based)

    while i < n:
        chunk_start = i + 1  # 1-based for output
        chunk_end = min(n, i + max_lines)  # 1-based

        # Try to find a symbol that starts within our current window
        best_symbol = None
        for sym in symbols:
            if sym.start >= chunk_start and sym.start <= chunk_end:
                # Check if the entire symbol fits within max_lines from current position
                symbol_size = sym.end - sym.start + 1
                if symbol_size <= max_lines and sym.end <= i + max_lines:
                    best_symbol = sym
                    break

        if best_symbol:
            # Chunk this complete symbol
            chunk_text = "\n".join(lines[best_symbol.start - 1 : best_symbol.end])
            chunks.append(
                {
                    "text": chunk_text,
                    "start": best_symbol.start,
                    "end": best_symbol.end,
                    "symbol": best_symbol.name,
                    "kind": best_symbol.kind,
                }
            )
            # Move past this symbol with minimal overlap
            i = max(best_symbol.end - overlap, i + 1)
        else:
            # No suitable symbol found, fall back to line-based chunking
            chunk_text = "\n".join(lines[i : i + max_lines])
            actual_end = min(n, i + max_lines)
            chunks.append({"text": chunk_text, "start": i + 1, "end": actual_end})
            i = max(actual_end - overlap, i + 1)

    return chunks


# --- Token-based micro-chunking (ReFRAG-lite) ---
# Produces tiny fixed-size token windows with stride, maps back to original line ranges.
def chunk_by_tokens(
    text: str, k_tokens: int = None, stride_tokens: int = None
) -> List[Dict]:
    try:
        from tokenizers import Tokenizer  # lightweight, already in requirements
    except Exception:
        Tokenizer = None  # type: ignore

    try:
        k = int(os.environ.get("MICRO_CHUNK_TOKENS", str(k_tokens or 16)) or 16)
    except Exception:
        k = 16
    try:
        s = int(
            os.environ.get("MICRO_CHUNK_STRIDE", str(stride_tokens or max(1, k // 2)))
            or max(1, k // 2)
        )
    except Exception:
        s = max(1, k // 2)

    # Helper: simple regex-based token offsets when HF tokenizer JSON is unavailable
    def _simple_offsets(txt: str):
        import re
        offs = []
        for m in re.finditer(r"\S+", txt):
            offs.append((m.start(), m.end()))
        return offs

    offsets = []
    # Load tokenizer; default to local model file if present
    tok_path = os.environ.get(
        "TOKENIZER_JSON", str((ROOT_DIR / "models" / "tokenizer.json"))
    )
    if Tokenizer is not None:
        try:
            tokenizer = Tokenizer.from_file(tok_path)
            try:
                enc = tokenizer.encode(text)
                offsets = getattr(enc, "offsets", None) or []
            except Exception:
                offsets = []
        except Exception:
            offsets = []

    if not offsets:
        # Fallback to simple regex tokenization; avoids degrading to 120-line chunks
        if os.environ.get("DEBUG_CHUNKING"):
            print("[ingest] tokenizers missing/unusable -> using simple regex tokenization")
        offsets = _simple_offsets(text)

    if not offsets:
        return chunk_lines(text, max_lines=120, overlap=20)

    # Precompute line starts for fast char->line mapping
    lines = text.splitlines(keepends=True)
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)
    total_chars = len(text)

    def char_to_line(c: int) -> int:
        # Binary search line_starts to find 1-based line number
        lo, hi = 0, len(line_starts) - 1
        if c <= 0:
            return 1
        if c >= total_chars:
            return len(lines)
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= c:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans + 1  # 1-based

    chunks: List[Dict] = []
    i = 0
    n = len(offsets)
    while i < n:
        j = min(n, i + k)
        start_char = offsets[i][0]
        end_char = offsets[j - 1][1] if j - 1 < n else offsets[-1][1]
        start_char = max(0, start_char)
        end_char = min(total_chars, max(start_char, end_char))
        chunk_text = text[start_char:end_char]
        if chunk_text:
            start_line = char_to_line(start_char)
            end_line = (
                char_to_line(end_char - 1) if end_char > start_char else start_line
            )
            chunks.append(
                {
                    "text": chunk_text,
                    "start": start_line,
                    "end": end_line,
                }
            )
        if j == n:
            break
        i = i + s if s > 0 else i + 1
    return chunks


from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.utils import lex_hash_vector_text as _lex_hash_vector_text


# Optional index-time pseudo descriptions for micro-chunks
# Enabled via REFRAG_PSEUDO_DESCRIBE=1 and requires REFRAG_DECODER=1

def _pseudo_describe_enabled() -> bool:
    try:
        return str(os.environ.get("REFRAG_PSEUDO_DESCRIBE", "0")).strip().lower() in {"1","true","yes","on"}
    except Exception:
        return False


def generate_pseudo_tags(text: str) -> tuple[str, list[str]]:
    """Best-effort: ask local decoder to produce a short label and 3-6 tags.
    Returns (pseudo, tags). On failure returns ("", [])."""
    pseudo: str = ""
    tags: list[str] = []
    if not _pseudo_describe_enabled() or not text.strip():
        return pseudo, tags
    try:
        from scripts.refrag_llamacpp import LlamaCppRefragClient, is_decoder_enabled  # type: ignore
        if not is_decoder_enabled():
            return "", []
        # Keep decoding tight/fast – this is only enrichment for retrieval
        prompt = (
            "You label code spans for search enrichment.\n"
            "Return strictly JSON: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
            "Code:\n" + text[:2000]
        )
        client = LlamaCppRefragClient()
        out = client.generate_with_soft_embeddings(
            prompt=prompt,
            max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
            temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
            top_k=int(os.environ.get("PSEUDO_TOP_K", "30") or 30),
            top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
            stop=["\n\n"],
        )
        import json as _json
        try:
            obj = _json.loads(out)
            if isinstance(obj, dict):
                p = obj.get("pseudo")
                t = obj.get("tags")
                if isinstance(p, str):
                    pseudo = p.strip()[:256]
                if isinstance(t, list):
                    tags = [str(x).strip() for x in t if str(x).strip()][:6]
        except Exception:
            pass
    except Exception:
        return "", []
    return pseudo, tags


def ensure_collection(client: QdrantClient, name: str, dim: int, vector_name: str):
    """Ensure collection exists with named vectors.
    Always includes dense (vector_name) and lexical (LEX_VECTOR_NAME).
    When REFRAG_MODE=1, also includes a compact mini vector (MINI_VECTOR_NAME).
    """
    try:
        info = client.get_collection(name)
        # Ensure HNSW tuned params even if the collection already existed
        try:
            client.update_collection(
                collection_name=name,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
            )
        except Exception:
            pass
        # Schema repair: add missing named vectors on existing collections
        try:
            cfg = getattr(info.config.params, "vectors", None)
            if isinstance(cfg, dict):
                missing = {}
                if LEX_VECTOR_NAME not in cfg:
                    missing[LEX_VECTOR_NAME] = models.VectorParams(
                        size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
                    )
                try:
                    refrag_on = os.environ.get("REFRAG_MODE", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                except Exception:
                    refrag_on = False
                if refrag_on and MINI_VECTOR_NAME not in cfg:
                    missing[MINI_VECTOR_NAME] = models.VectorParams(
                        size=int(
                            os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM
                        ),
                        distance=models.Distance.COSINE,
                    )
                if missing:
                    try:
                        client.update_collection(
                            collection_name=name, vectors_config=missing
                        )
                    except Exception:
                        # Best-effort; if server doesn't support adding vectors, leave to recreate path
                        pass
        except Exception:
            pass
        return
    except Exception:
        pass
    vectors_cfg = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(
            size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    # Conditionally add mini vector for ReFRAG gating
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            vectors_cfg[MINI_VECTOR_NAME] = models.VectorParams(
                size=int(os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM),
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )


def recreate_collection(client: QdrantClient, name: str, dim: int, vector_name: str):
    """Drop and recreate collection with named vectors (dense + lex [+ mini when REFRAG_MODE=1])."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    vectors_cfg = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(
            size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            vectors_cfg[MINI_VECTOR_NAME] = models.VectorParams(
                size=int(os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM),
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )


def ensure_payload_indexes(client: QdrantClient, collection: str):
    """Create helpful payload indexes if they don't exist (idempotent)."""
    for field in (
        "metadata.language",
        "metadata.path_prefix",
        "metadata.repo",
        "metadata.kind",
        "metadata.symbol",
        "metadata.symbol_path",
        "metadata.imports",
        "metadata.calls",
        "metadata.file_hash",
        "metadata.ingested_at",
        "metadata.last_modified_at",
        "metadata.churn_count",
        "metadata.author_count",
        "pid_str",
    ):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


# Lightweight import extraction per language (best-effort)
def _extract_imports(language: str, text: str) -> list:
    lines = text.splitlines()
    imps = []
    if language == "python":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.]+)", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*from\s+([\w\.]+)\s+import\s+", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language in ("javascript", "typescript"):
        for ln in lines:
            m = re.match(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*require\(\s*['\"]([^'\"]+)['\"]\s*\)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "go":
        block = False
        for ln in lines:
            if re.match(r"^\s*import\s*\(", ln):
                block = True
                continue
            if block:
                if ")" in ln:
                    block = False
                    continue
                m = re.match(r"^\s*\"([^\"]+)\"", ln)
                if m:
                    imps.append(m.group(1))
                    continue
            m = re.match(r"^\s*import\s+\"([^\"]+)\"", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "java":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.\*]+);", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "csharp":
        for ln in lines:
            # using Namespace.Sub; using static System.Math; using Alias = Namespace.Type;
            m = re.match(r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\._]*)(?:\s*;|\s*=)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "php":
        for ln in lines:
            # Namespaced uses: use Foo\Bar; use function Foo\bar; use const Foo\BAR;
            m = re.match(r"^\s*use\s+(?:function\s+|const\s+)?([A-Za-z_][A-Za-z0-9_\\\\]*)\s*;", ln)
            if m:
                imps.append(m.group(1).replace("\\\\", "\\"))
                continue
        for ln in lines:
            # include/require path-like imports
            m = re.match(r"^\s*(?:include|include_once|require|require_once)\s*\(?\s*['\"]([^'\"]+)['\"]\s*\)?\s*;", ln)
            if m:
                imps.append(m.group(1))
                continue

    elif language == "rust":
        for ln in lines:
            m = re.match(r"^\s*use\s+([^;]+);", ln)
            if m:
                imps.append(m.group(1).strip())
                continue
    elif language == "terraform":
        # modules/providers are most relevant cross-file references
        for ln in lines:
            m = re.match(r"^\s*source\s*=\s*['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*provider\s*=\s*['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "powershell":
        for ln in lines:
            m = re.match(
                r"^\s*Import-Module\s+([A-Za-z0-9_.\-]+)", ln, flags=re.IGNORECASE
            )
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*using\s+module\s+([^\s;]+)", ln, flags=re.IGNORECASE)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*using\s+namespace\s+([^\s;]+)", ln, flags=re.IGNORECASE)
            if m:
                imps.append(m.group(1))
                continue
    return imps[:200]


# Lightweight call-site extraction (best-effort, language-agnostic heuristics)
def _extract_calls(language: str, text: str) -> list:
    names = []
    # Simple heuristic: word followed by '(' that isn't a keyword
    kw = set(
        [
            "if",
            "for",
            "while",
            "switch",
            "return",
            "new",
            "catch",
            "func",
            "def",
            "class",
            "match",
        ]
    )
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = m.group(1)
        if name not in kw:
            names.append(name)
    # Deduplicate preserving order
    out = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out[:200]


def get_indexed_file_hash(client: QdrantClient, collection: str, file_path: str) -> str:
    """Return previously indexed file hash for this path, or empty string."""
    try:
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path", match=models.MatchValue(value=file_path)
                )
            ]
        )
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            with_payload=True,
            limit=1,
        )
        if points:
            md = (points[0].payload or {}).get("metadata") or {}
            return str(md.get("file_hash") or "")
    except Exception:
        return ""
    return ""


def delete_points_by_path(client: QdrantClient, collection: str, file_path: str):
    try:
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path", match=models.MatchValue(value=file_path)
                )
            ]
        )
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=filt),
            wait=True,
        )
    except Exception:
        pass


def embed_batch(model: TextEmbedding, texts: List[str]) -> List[List[float]]:
    # fastembed returns a generator of numpy arrays
    return [vec.tolist() for vec in model.embed(texts)]


def upsert_points(
    client: QdrantClient, collection: str, points: List[models.PointStruct]
):
    if not points:
        return
    # Safer upsert for large payloads: chunk + retry with backoff
    try:
        bsz = int(os.environ.get("INDEX_UPSERT_BATCH", "256") or 256)
    except Exception:
        bsz = 256
    try:
        retries = int(os.environ.get("INDEX_UPSERT_RETRIES", "3") or 3)
    except Exception:
        retries = 3
    try:
        backoff = float(os.environ.get("INDEX_UPSERT_BACKOFF", "0.5") or 0.5)
    except Exception:
        backoff = 0.5

    for i in range(0, len(points), max(1, bsz)):
        batch = points[i : i + max(1, bsz)]
        attempt = 0
        while True:
            try:
                client.upsert(collection_name=collection, points=batch, wait=True)
                break
            except Exception:
                attempt += 1
                if attempt >= retries:
                    # Last-resort: try smaller sub-batches to avoid dropping updates entirely
                    sub_size = max(1, bsz // 4)
                    for j in range(0, len(batch), sub_size):
                        sub = batch[j : j + sub_size]
                        try:
                            client.upsert(
                                collection_name=collection, points=sub, wait=True
                            )
                        except Exception:
                            # Give up on this tiny sub-batch; continue with the rest
                            pass
                    break
                else:
                    try:
                        time.sleep(backoff * attempt)
                    except Exception:
                        pass


def detect_language(path: Path) -> str:
    return CODE_EXTS.get(path.suffix.lower(), "unknown")


def build_information(
    language: str, path: Path, start: int, end: int, first_line: str
) -> str:
    first_line = (first_line or "").strip()
    if len(first_line) > 160:
        first_line = first_line[:160] + "…"
    return f"{language} code from {path} lines {start}-{end}. {first_line}"


# ---- Symbol extraction helpers ----
class _Sym(dict):
    __getattr__ = dict.get


def _extract_symbols_python(text: str) -> List[_Sym]:
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    out: List[_Sym] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(
                _Sym(
                    kind="function",
                    name=node.name,
                    start=getattr(node, "lineno", 0),
                    end=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                )
            )
        elif isinstance(node, ast.ClassDef):
            out.append(
                _Sym(
                    kind="class",
                    name=node.name,
                    start=getattr(node, "lineno", 0),
                    end=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                )
            )
    # Filter invalid
    return [s for s in out if s.start and s.end and s.end >= s.start]


_JS_FUNC_PATTERNS = [
    r"^\s*export\s+function\s+([A-Za-z_$][\w$]*)\s*\(",
    r"^\s*function\s+([A-Za-z_$][\w$]*)\s*\(",
    r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>",
    r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*function\s*\(",
]
_JS_CLASS_PATTERNS = [r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\b"]


def _extract_symbols_js_like(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        for pat in _JS_CLASS_PATTERNS:
            m = re.match(pat, line)
            if m:
                syms.append(_Sym(kind="class", name=m.group(1), start=idx, end=idx))
                break
        for pat in _JS_FUNC_PATTERNS:
            m = re.match(pat, line)
            if m:
                syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
                break
    # Approximate end by next symbol start-1
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        if i + 1 < len(syms):
            syms[i]["end"] = max(syms[i].start, syms[i + 1].start - 1)
        else:
            syms[i]["end"] = max(syms[i].start, len(lines))
    return syms


def _extract_symbols_go(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*type\s+([A-Za-z_][\w]*)\s+struct\b", line)
        if m:
            syms.append(_Sym(kind="struct", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*type\s+([A-Za-z_][\w]*)\s+interface\b", line)
        if m:
            syms.append(_Sym(kind="interface", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(
            r"^\s*func\s*\(\s*[^)]+\s+\*?([A-Za-z_][\w]*)\s*\)\s*([A-Za-z_][\w]*)\s*\(",
            line,
        )
        if m:
            syms.append(
                _Sym(
                    kind="method",
                    name=m.group(2),
                    path=f"{m.group(1)}.{m.group(2)}",
                    start=idx,
                    end=idx,
                )
            )
            continue
        m = re.match(r"^\s*func\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols_java(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_class = None
    for idx, line in enumerate(lines, 1):
        m = re.match(
            r"^\s*(?:public|protected|private)?\s*(?:final\s+|abstract\s+)?class\s+([A-Za-z_][\w]*)\b",
            line,
        )
        if m:
            current_class = m.group(1)
            syms.append(_Sym(kind="class", name=current_class, start=idx, end=idx))
            continue
        m = re.match(
            r"^\s*(?:public|protected|private)?\s*(?:static\s+)?[A-Za-z_<>,\[\]]+\s+([A-Za-z_][\w]*)\s*\(",
            line,
        )
        if m:
            name = m.group(1)
            path = f"{current_class}.{name}" if current_class else name
            syms.append(_Sym(kind="method", name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms



def _extract_symbols_csharp(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_type = None
    for idx, line in enumerate(lines, 1):
        # class / interface / struct / enum
        m = re.match(r"^\s*(?:public|protected|private|internal)?\s*(?:abstract\s+|sealed\s+|static\s+)?(class|interface|struct|enum)\s+([A-Za-z_][\w]*)\b", line)
        if m:
            kind, name = m.group(1), m.group(2)
            current_type = name
            kind_map = {"class": "class", "interface": "interface", "struct": "struct", "enum": "enum"}
            syms.append(_Sym(kind=kind_map.get(kind, "type"), name=name, start=idx, end=idx))
            continue
        # method (very heuristic)
        m = re.match(r"^\s*(?:public|protected|private|internal)?\s*(?:static\s+|virtual\s+|override\s+|async\s+)?[A-Za-z_<>,\[\]\.]+\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            name = m.group(1)
            path = f"{current_type}.{name}" if current_type else name
            syms.append(_Sym(kind="method", name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols_php(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_type = None
    depth = 0
    for idx, line in enumerate(lines, 1):
        # track simple brace depth to reset current_type when leaving class
        depth += line.count("{")
        depth -= line.count("}")
        if depth <= 0:
            current_type = None
        # namespace declaration (optional informational anchor)
        m = re.match(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_\\\\]*)\s*;", line)
        if m:
            ns = m.group(1).replace("\\\\", "\\")
            syms.append(_Sym(kind="namespace", name=ns, start=idx, end=idx))
            continue
        # class/interface/trait
        m = re.match(r"^\s*(?:final\s+|abstract\s+)?(class|interface|trait)\s+([A-Za-z_][\w]*)\b", line)
        if m:
            kind, name = m.group(1), m.group(2)
            current_type = name
            syms.append(_Sym(kind=kind, name=name, start=idx, end=idx))
            continue
        # methods or functions
        m = re.match(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?function\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            name = m.group(1)
            path = f"{current_type}.{name}" if current_type else name
            syms.append(_Sym(kind="method" if current_type else "function", name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms



def _extract_symbols_shell(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*([A-Za-z_][\w]*)\s*\(\)\s*\{", line)
        if m:
            syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*function\s+([A-Za-z_][\w]*)\s*\{", line)
        if m:
            syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
            continue
    return syms


def _extract_symbols_yaml(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        # treat Markdown-style headings in YAML files as anchors
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            syms.append(
                _Sym(kind="heading", name=m.group(1).strip(), start=idx, end=idx)
            )
    return syms


def _extract_symbols_powershell(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        if re.match(
            r"^\s*function\s+([A-Za-z_][\w-]*)\s*\{", line, flags=re.IGNORECASE
        ):
            name = (
                re.sub(r"^\s*function\s+", "", line, flags=re.IGNORECASE)
                .split("{")[0]
                .strip()
            )
            syms.append(_Sym(kind="function", name=name, start=idx, end=idx))
            continue
        m = re.match(r"^\s*class\s+([A-Za-z_][\w-]*)\s*\{", line, flags=re.IGNORECASE)
        if m:
            syms.append(_Sym(kind="class", name=m.group(1), start=idx, end=idx))
            continue
    return syms


def _extract_symbols_rust(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_impl = None
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*impl(?:\s*<[^>]+>)?\s*([A-Za-z_][\w:]*)", line)
        if m:
            current_impl = m.group(1)
            syms.append(_Sym(kind="impl", name=current_impl, start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][\w]*)\b", line)
        if m:
            syms.append(_Sym(kind="struct", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][\w]*)\b", line)
        if m:
            syms.append(_Sym(kind="enum", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][\w]*)\b", line)
        if m:
            syms.append(_Sym(kind="trait", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            name = m.group(1)
            path = f"{current_impl}::{name}" if current_impl else name
            kind = "method" if current_impl else "function"
            syms.append(_Sym(kind=kind, name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


# ----- Tree-sitter based extraction (currently Python, JS/TS) -----
def _ts_parser(lang_key: str):
    if not _use_tree_sitter():
        return None
    try:
        p = Parser()
        p.set_language(get_language(lang_key))
        return p
    except Exception:
        return None


def _ts_extract_symbols_python(text: str) -> List[_Sym]:
    parser = _ts_parser("python")
    if not parser:
        return []
    tree = parser.parse(text.encode("utf-8"))
    root = tree.root_node
    syms: List[_Sym] = []

    def node_text(n):
        return text.encode("utf-8")[n.start_byte : n.end_byte].decode(
            "utf-8", errors="ignore"
        )

    class_stack: List[str] = []

    def walk(n):
        t = n.type
        if t == "class_definition":
            name_node = n.child_by_field_name("name")
            cls = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            syms.append(_Sym(kind="class", name=cls, start=start, end=end))
            class_stack.append(cls)
            # Walk body
            for c in n.children:
                walk(c)
            class_stack.pop()
            return
        if t == "function_definition":
            name_node = n.child_by_field_name("name")
            fn = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            if class_stack:
                path = f"{class_stack[-1]}.{fn}"
                syms.append(
                    _Sym(kind="method", name=fn, path=path, start=start, end=end)
                )
            else:
                syms.append(_Sym(kind="function", name=fn, start=start, end=end))
        for c in n.children:
            walk(c)

    walk(root)
    return syms


def _ts_extract_symbols_js(text: str) -> List[_Sym]:
    # Works for javascript/typescript using a generic JS parser
    parser = _ts_parser("javascript")
    if not parser:
        return []
    tree = parser.parse(text.encode("utf-8"))
    root = tree.root_node
    syms: List[_Sym] = []

    def node_text(n):
        return text.encode("utf-8")[n.start_byte : n.end_byte].decode(
            "utf-8", errors="ignore"
        )

    class_stack: List[str] = []

    def walk(n):
        t = n.type
        if t == "class_declaration":
            name_node = n.child_by_field_name("name")
            cls = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            syms.append(_Sym(kind="class", name=cls, start=start, end=end))
            class_stack.append(cls)
            for c in n.children:
                walk(c)
            class_stack.pop()
            return
        if t in ("function_declaration",):
            name_node = n.child_by_field_name("name")
            fn = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            syms.append(_Sym(kind="function", name=fn, start=start, end=end))
        if t == "method_definition":
            name_node = n.child_by_field_name("name")
            m = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            path = f"{class_stack[-1]}.{m}" if class_stack else m
            syms.append(_Sym(kind="method", name=m, path=path, start=start, end=end))
        for c in n.children:
            walk(c)

    walk(root)
    return syms


def _ts_extract_symbols(language: str, text: str) -> List[_Sym]:
    if language == "python":
        return _ts_extract_symbols_python(text)
    if language in ("javascript", "typescript"):
        return _ts_extract_symbols_js(text)
    return []


def _ts_extract_imports_calls_python(text: str):
    parser = _ts_parser("python")
    if not parser:
        return [], []
    data = text.encode("utf-8")
    tree = parser.parse(data)
    root = tree.root_node

    def node_text(n):
        return data[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")

    imports: List[str] = []
    calls: List[str] = []

    def walk(n):
        t = n.type
        if t == "import_statement":
            s = node_text(n)
            m = re.search(r"\bimport\s+([\w\.]+)", s)
            if m:
                imports.append(m.group(1))
        elif t == "import_from_statement":
            s = node_text(n)
            m = re.search(r"\bfrom\s+([\w\.]+)\s+import\b", s)
            if m:
                imports.append(m.group(1))
        elif t == "call":
            func = n.child_by_field_name("function")
            if func:
                name = node_text(func)
                # Take the last attribute part if dotted
                base = re.split(r"[\.:]", name)[-1]
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base):
                    calls.append(base)
        for c in n.children:
            walk(c)

    walk(root)
    # Deduplicate preserving order
    seen = set()
    calls_dedup = []
    for x in calls:
        if x not in seen:
            calls_dedup.append(x)
            seen.add(x)
    return imports[:200], calls_dedup[:200]


def _get_imports_calls(language: str, text: str):
    if _use_tree_sitter() and language == "python":
        return _ts_extract_imports_calls_python(text)
    return _extract_imports(language, text), _extract_calls(language, text)


def _extract_symbols_terraform(text: str) -> List[_Sym]:
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*(resource)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s*\{", line)
        if m:
            t, name = m.group(2), m.group(3)
            syms.append(
                _Sym(kind="resource", name=name, path=f"{t}.{name}", start=idx, end=idx)
            )
            continue
        m = re.match(r"^\s*(data)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s*\{", line)
        if m:
            t, name = m.group(2), m.group(3)
            syms.append(
                _Sym(
                    kind="data", name=name, path=f"data.{t}.{name}", start=idx, end=idx
                )
            )
            continue
        m = re.match(r"^\s*(module)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(
                    kind="module", name=name, path=f"module.{name}", start=idx, end=idx
                )
            )
            continue
        m = re.match(r"^\s*(variable)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(kind="variable", name=name, path=f"var.{name}", start=idx, end=idx)
            )
            continue
        m = re.match(r"^\s*(output)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(
                    kind="output", name=name, path=f"output.{name}", start=idx, end=idx
                )
            )
            continue
        m = re.match(r"^\s*(provider)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(
                    kind="provider",
                    name=name,
                    path=f"provider.{name}",
                    start=idx,
                    end=idx,
                )
            )
            continue
        m = re.match(r"^\s*(locals)\s*\{", line)
        if m:
            syms.append(
                _Sym(kind="locals", name="locals", path="locals", start=idx, end=idx)
            )
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols(language: str, text: str) -> List[_Sym]:
    # Prefer tree-sitter when enabled and supported; fallback to existing extractors
    if _use_tree_sitter():
        ts_syms = _ts_extract_symbols(language, text)
        if ts_syms:
            return ts_syms
    if language == "python":
        return _extract_symbols_python(text)
    if language in ("javascript", "typescript"):
        return _extract_symbols_js_like(text)
    if language == "go":
        return _extract_symbols_go(text)
    if language == "java":
        return _extract_symbols_java(text)
    if language == "rust":
        return _extract_symbols_rust(text)
    if language == "terraform":
        return _extract_symbols_terraform(text)
    if language == "shell":
        return _extract_symbols_shell(text)
    if language == "yaml":
        return _extract_symbols_yaml(text)
    if language == "powershell":
        return _extract_symbols_powershell(text)
    if language == "csharp":
        return _extract_symbols_csharp(text)
    if language == "php":
        return _extract_symbols_php(text)
    return []


def _choose_symbol_for_chunk(start: int, end: int, symbols: List[_Sym]):
    if not symbols:
        return "", "", ""
    overlaps = [s for s in symbols if s.start <= end and s.end >= start]

    def pick(sym):
        name = sym.get("name") or ""
        path = sym.get("path") or name
        return sym.get("kind") or "", name, path

    if overlaps:
        overlaps.sort(key=lambda s: (-(s.start), (s.end - s.start)))
        return pick(overlaps[0])
    preceding = [s for s in symbols if s.start <= end]
    if preceding:
        s = max(preceding, key=lambda x: x.start)
        return pick(s)
    return "", "", ""


def index_single_file(
    client: QdrantClient,
    model: TextEmbedding,
    collection: str,
    vector_name: str,
    file_path: Path,
    *,
    dedupe: bool = True,
    skip_unchanged: bool = True,
) -> bool:
    """Index a single file path. Returns True if indexed, False if skipped."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return False

    language = detect_language(file_path)
    file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    repo_tag = _detect_repo_name_from_path(file_path)

    if skip_unchanged:
        prev = get_indexed_file_hash(client, collection, str(file_path))
        if prev and prev == file_hash:
            print(f"Skipping unchanged file: {file_path}")
            return False

    if dedupe:
        delete_points_by_path(client, collection, str(file_path))

    symbols = _extract_symbols(language, text)
    imports, calls = _get_imports_calls(language, text)
    last_mod, churn_count, author_count = _git_metadata(file_path)

    CHUNK_LINES = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    CHUNK_OVERLAP = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)
    # Micro-chunking (token-based) takes precedence; else semantic; else line-based
    use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_micro:
        chunks = chunk_by_tokens(text)
        try:
            _cap = int(os.environ.get("MAX_MICRO_CHUNKS_PER_FILE", "500") or 500)
            if _cap > 0 and len(chunks) > _cap:
                _before = len(chunks)
                chunks = chunks[:_cap]
                try:
                    print(
                        f"[ingest] micro-chunks capped path={file_path} count={_before}->{len(chunks)} cap={_cap}"
                    )
                except Exception:
                    pass
        except Exception:
            pass
    elif use_semantic:
        chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
    else:
        chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)
    batch_texts: List[str] = []
    batch_meta: List[Dict] = []
    batch_ids: List[int] = []
    batch_lex: List[list[float]] = []

    def make_point(pid, dense_vec, lex_vec, payload):
        if vector_name:
            vecs = {vector_name: dense_vec, LEX_VECTOR_NAME: lex_vec}
            try:
                if os.environ.get("REFRAG_MODE", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    vecs[MINI_VECTOR_NAME] = project_mini(list(dense_vec), MINI_VEC_DIM)
            except Exception:
                pass
            return models.PointStruct(id=pid, vector=vecs, payload=payload)
        else:
            # unnamed collection: store dense only
            return models.PointStruct(id=pid, vector=dense_vec, payload=payload)

    for ch in chunks:
        info = build_information(
            language,
            file_path,
            ch["start"],
            ch["end"],
            ch["text"].splitlines()[0] if ch["text"] else "",
        )
        kind, sym, sym_path = _choose_symbol_for_chunk(ch["start"], ch["end"], symbols)
        # Prefer embedded symbol metadata from semantic chunker when present
        if "kind" in ch and ch.get("kind"):
            kind = ch.get("kind") or kind
        if "symbol" in ch and ch.get("symbol"):
            sym = ch.get("symbol") or sym
        if "symbol_path" in ch and ch.get("symbol_path"):
            sym_path = ch.get("symbol_path") or sym_path

        # Track both container path (/work mirror) and original host path for clarity across environments
        _cur_path = str(file_path)
        _host_root = str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
        _host_path = None
        _container_path = None
        try:
            if _cur_path.startswith("/work/") and _host_root:
                _rel = _cur_path[len("/work/"):]
                _host_path = os.path.realpath(os.path.join(_host_root, _rel))
                _container_path = _cur_path
            else:
                # Likely indexing on the host directly
                _host_path = _cur_path
                if _host_root and _cur_path.startswith((_host_root + "/")):
                    _rel = _cur_path[len(_host_root) + 1 :]
                    _container_path = "/work/" + _rel
        except Exception:
            _host_path = _cur_path
            _container_path = _cur_path if _cur_path.startswith("/work/") else None

        payload = {
            "document": info,
            "information": info,
            "metadata": {
                "path": str(file_path),
                "path_prefix": str(file_path.parent),
                "language": language,
                "kind": kind,
                "symbol": sym,
                "symbol_path": sym_path,
                "repo": repo_tag,
                "start_line": ch["start"],
                "end_line": ch["end"],
                "code": ch["text"],
                "file_hash": file_hash,
                "imports": imports,
                "calls": calls,
                "ingested_at": int(time.time()),
                "last_modified_at": int(last_mod),
                "churn_count": int(churn_count),
                "author_count": int(author_count),
                # New: explicit dual-path tracking
                "host_path": _host_path,
                "container_path": _container_path,
            },
        }
        # Optional LLM enrichment for lexical retrieval: pseudo + tags per micro-chunk
        pseudo, tags = ("", [])
        try:
            pseudo, tags = generate_pseudo_tags(ch.get("text") or "")
            if pseudo:
                payload["pseudo"] = pseudo
            if tags:
                payload["tags"] = tags
        except Exception:
            pass
        batch_texts.append(info)
        batch_meta.append(payload)
        batch_ids.append(hash_id(ch["text"], str(file_path), ch["start"], ch["end"]))
        aug_lex_text = (ch.get("text") or "") + (" " + pseudo if pseudo else "") + (" " + " ".join(tags) if tags else "")
        batch_lex.append(_lex_hash_vector_text(aug_lex_text))

    if batch_texts:
        vectors = embed_batch(model, batch_texts)
        # Inject pid_str into payloads for server-side gating
        for _idx, _m in enumerate(batch_meta):
            try:
                _m["pid_str"] = str(batch_ids[_idx])
            except Exception:
                pass
        points = [
            make_point(i, v, lx, m)
            for i, v, lx, m in zip(batch_ids, vectors, batch_lex, batch_meta)
        ]
        upsert_points(client, collection, points)
        return True
    return False


def index_repo(
    root: Path,
    qdrant_url: str,
    api_key: str,
    collection: str,
    model_name: str,
    recreate: bool,
    *,
    dedupe: bool = True,
    skip_unchanged: bool = True,
):
    print(
        f"Indexing root={root} -> {qdrant_url} collection={collection} model={model_name} recreate={recreate}"
    )
    model = TextEmbedding(model_name=model_name)
    # Determine embedding dimension
    dim = len(next(model.embed(["dimension probe"])))

    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key or None,
        timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
    )

    # Determine vector name
    if recreate:
        vector_name = _sanitize_vector_name(model_name)
    else:
        vector_name = None
        try:
            info = client.get_collection(collection)
            cfg = info.config.params.vectors
            if isinstance(cfg, dict) and cfg:
                # Prefer named vector whose size matches current embedding dim
                for name, params in cfg.items():
                    psize = getattr(params, "size", None) or getattr(
                        params, "dim", None
                    )
                    if psize and int(psize) == int(dim):
                        vector_name = name
                        break
                # Otherwise, if a LEX vector exists, pick a different name as dense
                if vector_name is None and LEX_VECTOR_NAME in cfg:
                    for name in cfg.keys():
                        if name != LEX_VECTOR_NAME:
                            vector_name = name
                            break
        except Exception:
            pass
        if vector_name is None:
            vector_name = _sanitize_vector_name(model_name)

    # Workspace state: announce indexing start
    try:
        if update_workspace_state:
            update_workspace_state("/work", {"qdrant_collection": collection})
        if update_indexing_status:
            update_indexing_status(
                "/work",
                {
                    "state": "indexing",
                    "started_at": datetime.now().isoformat(),
                    "progress": {"files_processed": 0, "total_files": None},
                },
            )
    except Exception:
        pass


    if recreate:
        recreate_collection(client, collection, dim, vector_name)
    else:
        ensure_collection(client, collection, dim, vector_name)

    # Ensure useful payload indexes exist (idempotent)
    ensure_payload_indexes(client, collection)
    # Repo tag for filtering: auto-detect from git or folder name
    repo_tag = _detect_repo_name_from_path(root)

    # Batch and scaling config (env/CLI overridable)
    batch_texts: list[str] = []
    batch_meta: list[dict] = []
    batch_ids: list[int] = []
    batch_lex: list[list[float]] = []
    BATCH_SIZE = int(os.environ.get("INDEX_BATCH_SIZE", "64") or 64)
    CHUNK_LINES = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    CHUNK_OVERLAP = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)
    PROGRESS_EVERY = int(os.environ.get("INDEX_PROGRESS_EVERY", "200") or 200)
    # Semantic chunking toggle
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Debug chunking mode
    if os.environ.get("DEBUG_CHUNKING"):
        print(f"[DEBUG] INDEX_SEMANTIC_CHUNKS={os.environ.get('INDEX_SEMANTIC_CHUNKS', 'NOT_SET')} -> use_semantic={use_semantic}")
        print(f"[DEBUG] INDEX_MICRO_CHUNKS={os.environ.get('INDEX_MICRO_CHUNKS', 'NOT_SET')}")

    files_seen = 0
    files_indexed = 0
    points_indexed = 0

    def make_point(pid, dense_vec, lex_vec, payload):
        # Use named vectors if collection has names: store dense + lexical (+ mini if REFRAG_MODE)
        if vector_name:
            vecs = {vector_name: dense_vec, LEX_VECTOR_NAME: lex_vec}
            try:
                if os.environ.get("REFRAG_MODE", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    vecs[MINI_VECTOR_NAME] = project_mini(list(dense_vec), MINI_VEC_DIM)
            except Exception:
                pass
            return models.PointStruct(id=pid, vector=vecs, payload=payload)
        else:
            # unnamed collection: store dense only
            return models.PointStruct(id=pid, vector=dense_vec, payload=payload)

    for file_path in iter_files(root):
        files_seen += 1
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue
        language = detect_language(file_path)
        file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

        # Skip unchanged files if enabled (default)
        if skip_unchanged:
            prev = get_indexed_file_hash(client, collection, str(file_path))
            if prev and prev == file_hash:
                if PROGRESS_EVERY <= 0 and files_seen % 50 == 0:
                    # minor heartbeat when no progress cadence configured
                    print(f"... processed {files_seen} files (skipping unchanged)")
                continue

        # Dedupe per-file by deleting previous points for this path (default)
        if dedupe:
            delete_points_by_path(client, collection, str(file_path))

        files_indexed += 1
        symbols = _extract_symbols(language, text)
        imports, calls = _get_imports_calls(language, text)
        last_mod, churn_count, author_count = _git_metadata(file_path)

        # Micro-chunking (token-based) takes precedence; else semantic; else line-based
        use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_micro:
            chunks = chunk_by_tokens(text)
            try:
                _cap = int(os.environ.get("MAX_MICRO_CHUNKS_PER_FILE", "500") or 500)
                if _cap > 0 and len(chunks) > _cap:
                    _before = len(chunks)
                    chunks = chunks[:_cap]
                    try:
                        print(
                            f"[ingest] micro-chunks capped path={file_path} count={_before}->{len(chunks)} cap={_cap}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        elif use_semantic:
            chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
        else:
            chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)
        for ch in chunks:
            info = build_information(
                language,
                file_path,
                ch["start"],
                ch["end"],
                ch["text"].splitlines()[0] if ch["text"] else "",
            )
            kind, sym, sym_path = _choose_symbol_for_chunk(
                ch["start"], ch["end"], symbols
            )
            # If chunk_semantic returned embedded symbol/kind, prefer it
            if "kind" in ch and ch.get("kind"):
                kind = ch.get("kind") or kind
            if "symbol" in ch and ch.get("symbol"):
                sym = ch.get("symbol") or sym
            if "symbol_path" in ch and ch.get("symbol_path"):
                sym_path = ch.get("symbol_path") or sym_path
            # Track both container path (/work mirror) and original host path
            _cur_path = str(file_path)
            _host_root = str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
            _host_path = None
            _container_path = None
            try:
                if _cur_path.startswith("/work/") and _host_root:
                    _rel = _cur_path[len("/work/"):]
                    _host_path = os.path.realpath(os.path.join(_host_root, _rel))
                    _container_path = _cur_path
                else:
                    _host_path = _cur_path
                    if _host_root and _cur_path.startswith((_host_root + "/")):
                        _rel = _cur_path[len(_host_root) + 1 :]
                        _container_path = "/work/" + _rel
            except Exception:
                _host_path = _cur_path
                _container_path = _cur_path if _cur_path.startswith("/work/") else None

            payload = {
                "document": info,
                "information": info,
                "metadata": {
                    "path": str(file_path),
                    "path_prefix": str(file_path.parent),
                    "language": language,
                    "kind": kind,
                    "symbol": sym,
                    "symbol_path": sym_path or "",
                    "repo": repo_tag,
                    "start_line": ch["start"],
                    "end_line": ch["end"],
                    "code": ch["text"],
                    "file_hash": file_hash,
                    "imports": imports,
                    "calls": calls,
                    "ingested_at": int(time.time()),
                    "last_modified_at": int(last_mod),
                    "churn_count": int(churn_count),
                    "author_count": int(author_count),
                    # New: dual-path tracking
                    "host_path": _host_path,
                    "container_path": _container_path,
                },
            }
            # Optional LLM enrichment for lexical retrieval: pseudo + tags per micro-chunk
            pseudo, tags = ("", [])
            try:
                pseudo, tags = generate_pseudo_tags(ch.get("text") or "")
                if pseudo:
                    payload["pseudo"] = pseudo
                if tags:
                    payload["tags"] = tags
            except Exception:
                pass
            batch_texts.append(info)
            batch_meta.append(payload)
            batch_ids.append(
                hash_id(ch["text"], str(file_path), ch["start"], ch["end"])
            )
            aug_lex_text = (ch.get("text") or "") + (" " + pseudo if pseudo else "") + (" " + " ".join(tags) if tags else "")
            batch_lex.append(_lex_hash_vector_text(aug_lex_text))
            points_indexed += 1
            if len(batch_texts) >= BATCH_SIZE:
                vectors = embed_batch(model, batch_texts)
                # Inject pid_str into payloads for server-side gating
                for _idx, _m in enumerate(batch_meta):
                    try:
                        _m["pid_str"] = str(batch_ids[_idx])
                    except Exception:
                        pass
                points = [
                    make_point(i, v, lx, m)
                    for i, v, lx, m in zip(batch_ids, vectors, batch_lex, batch_meta)
                ]
                upsert_points(client, collection, points)
                batch_texts, batch_meta, batch_ids, batch_lex = [], [], [], []

        if PROGRESS_EVERY > 0 and files_seen % PROGRESS_EVERY == 0:
            print(
                f"Progress: files_seen={files_seen}, files_indexed={files_indexed}, chunks_indexed={points_indexed}"
            )

    if batch_texts:
        vectors = embed_batch(model, batch_texts)
        # Inject pid_str into payloads for server-side gating (final batch)
        for _idx, _m in enumerate(batch_meta):
            try:
                _m["pid_str"] = str(batch_ids[_idx])
            except Exception:
                pass
        points = [
            make_point(i, v, lx, m)
            for i, v, lx, m in zip(batch_ids, vectors, batch_lex, batch_meta)
        ]
        upsert_points(client, collection, points)

    print(
        f"Indexing complete. files_seen={files_seen}, files_indexed={files_indexed}, chunks_indexed={points_indexed}"
    )

    # Workspace state: mark completion
    try:
        if update_last_activity:
            update_last_activity(
                "/work",
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "scan-completed",
                    "filePath": "",
                    "details": {
                        "files_seen": files_seen,
                        "files_indexed": files_indexed,
                        "chunks_indexed": points_indexed,
                    },
                },
            )
        if update_indexing_status:
            update_indexing_status(
                "/work",
                {
                    "state": "idle",
                    "progress": {"files_processed": files_indexed, "total_files": None},
                },
            )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Index code into Qdrant with metadata for MCP code search."
    )
    parser.add_argument("--root", type=str, default=".", help="Root directory to index")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection before indexing",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not delete existing points for each file before inserting",
    )
    parser.add_argument(
        "--no-skip-unchanged",
        action="store_true",
        help="Do not skip files whose content hash matches existing index",
    )
    # Exclusion controls
    parser.add_argument(
        "--ignore-file",
        type=str,
        default=None,
        help="Path to a .qdrantignore-style file of patterns to exclude",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Disable default exclusions (models, node_modules, build, venv, .git, etc.)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Additional exclude pattern(s); can be used multiple times or comma-separated",
    )
    # Scaling controls
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding/upsert batch size (default 64)",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=None,
        help="Max lines per chunk (default 120)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap lines between chunks (default 20)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print progress every N files (default 200; 0 disables)",
    )

    args = parser.parse_args()

    # Map CLI overrides to env so downstream helpers pick them up
    if args.ignore_file:
        os.environ["QDRANT_IGNORE_FILE"] = args.ignore_file
    if args.no_default_excludes:
        os.environ["QDRANT_DEFAULT_EXCLUDES"] = "0"
    if args.exclude:
        # allow comma-separated and repeated flags
        parts = []
        for e in args.exclude:
            parts.extend([p.strip() for p in str(e).split(",") if p.strip()])
        if parts:
            os.environ["QDRANT_EXCLUDES"] = ",".join(parts)
    if args.batch_size is not None:
        os.environ["INDEX_BATCH_SIZE"] = str(args.batch_size)
    if args.chunk_lines is not None:
        os.environ["INDEX_CHUNK_LINES"] = str(args.chunk_lines)
    if args.chunk_overlap is not None:
        os.environ["INDEX_CHUNK_OVERLAP"] = str(args.chunk_overlap)
    if args.progress_every is not None:
        os.environ["INDEX_PROGRESS_EVERY"] = str(args.progress_every)

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    collection = os.environ.get("COLLECTION_NAME", "my-collection")
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    index_repo(
        Path(args.root).resolve(),
        qdrant_url,
        api_key,
        collection,
        model_name,
        args.recreate,
        dedupe=(not args.no_dedupe),
        skip_unchanged=(not args.no_skip_unchanged),
    )


if __name__ == "__main__":
    main()
