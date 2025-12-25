#!/usr/bin/env python3
"""
ingest_code.py - Façade module for code indexing.

This is the stable public entrypoint for the code indexing subsystem.
All internal logic has been refactored into smaller, focused modules in scripts/ingest/:
- config.py: Environment-based configuration and constants
- tree_sitter.py: Tree-sitter setup and language loading
- vectors.py: Vector generation utilities (lex hash, mini projection)
- exclusions.py: File and directory exclusion logic
- chunking.py: Code chunking utilities (line, semantic, token-based)
- symbols.py: Symbol extraction for code analysis
- pseudo.py: ReFRAG pseudo-description and tag generation
- metadata.py: Metadata extraction (git, imports, calls)
- qdrant.py: Qdrant schema and I/O operations
- pipeline.py: Helper functions for indexing
- cli.py: Command-line interface

This façade:
1. Re-exports all public APIs for backwards compatibility
2. Implements main orchestration functions (index_single_file, index_repo, process_file_with_smart_reindexing)
3. Provides the CLI entrypoint (main)
"""
from __future__ import annotations

import os
import sys
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

# Ensure project root is on sys.path when run as a script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# Re-exports from ingest/config.py
# ---------------------------------------------------------------------------
from scripts.ingest.config import (
    ROOT_DIR,
    _safe_int_env,
    _env_truthy,
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    MINI_VECTOR_NAME,
    MINI_VEC_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
    _STOP,
    CODE_EXTS,
    EXTENSIONLESS_FILES,
    _DEFAULT_EXCLUDE_DIRS,
    _DEFAULT_EXCLUDE_DIR_GLOBS,
    _DEFAULT_EXCLUDE_FILES,
    _ANY_DEPTH_EXCLUDE_DIR_NAMES,
    is_multi_repo_mode,
    get_collection_name,
    logical_repo_reuse_enabled,
    log_activity,
    get_cached_file_hash,
    set_cached_file_hash,
    remove_cached_file,
    update_indexing_status,
    update_workspace_state,
    get_cached_symbols,
    set_cached_symbols,
    remove_cached_symbols,
    compare_symbol_changes,
    get_cached_pseudo,
    set_cached_pseudo,
    update_symbols_with_pseudo,
    get_workspace_state,
    get_cached_file_meta,
    indexing_lock,
    file_indexing_lock,
    is_file_locked,
    _detect_repo_for_file,
    _get_collection_for_file,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/tree_sitter.py
# ---------------------------------------------------------------------------
from scripts.ingest.tree_sitter import (
    _TS_LANGUAGES,
    _TS_AVAILABLE,
    _use_tree_sitter,
    _ts_parser,
    _load_ts_language,
)

try:
    from tree_sitter import Parser, Language
except ImportError:
    Parser = None  # type: ignore
    Language = None  # type: ignore

# ---------------------------------------------------------------------------
# Re-exports from ingest/vectors.py
# ---------------------------------------------------------------------------
from scripts.ingest.vectors import (
    _MINI_PROJ_CACHE,
    _get_mini_proj,
    project_mini,
    _split_ident_lex,
    _lex_hash_vector,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/exclusions.py
# ---------------------------------------------------------------------------
from scripts.ingest.exclusions import (
    _Excluder,
    is_indexable_file,
    _is_indexable_file,
    _should_skip_explicit_file_by_excluder,
    iter_files,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/chunking.py
# ---------------------------------------------------------------------------
from scripts.ingest.chunking import (
    chunk_lines,
    chunk_semantic,
    chunk_by_tokens,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/symbols.py
# ---------------------------------------------------------------------------
from scripts.ingest.symbols import (
    _Sym,
    _extract_symbols_python,
    _extract_symbols_js_like,
    _extract_symbols_go,
    _extract_symbols_java,
    _extract_symbols_csharp,
    _extract_symbols_php,
    _extract_symbols_shell,
    _extract_symbols_yaml,
    _extract_symbols_powershell,
    _extract_symbols_rust,
    _extract_symbols_terraform,
    _ts_extract_symbols_python,
    _ts_extract_symbols_js,
    _ts_extract_symbols_yaml,
    _ts_extract_symbols,
    _extract_symbols,
    _choose_symbol_for_chunk,
    extract_symbols_with_tree_sitter,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/pseudo.py
# ---------------------------------------------------------------------------
from scripts.ingest.pseudo import (
    _pseudo_describe_enabled,
    _smart_symbol_reindexing_enabled,
    generate_pseudo_tags,
    should_process_pseudo_for_chunk,
    should_use_smart_reindexing,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/metadata.py
# ---------------------------------------------------------------------------
from scripts.ingest.metadata import (
    _git_metadata,
    _extract_imports,
    _extract_calls,
    _get_imports_calls,
    _get_host_path_from_origin,
    _compute_host_and_container_paths,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/qdrant.py
# ---------------------------------------------------------------------------
from scripts.ingest.qdrant import (
    ENSURED_COLLECTIONS,
    ENSURED_COLLECTIONS_LAST_CHECK,
    CollectionNeedsRecreateError,
    ensure_collection,
    recreate_collection,
    ensure_payload_indexes,
    ensure_collection_and_indexes_once,
    get_indexed_file_hash,
    delete_points_by_path,
    upsert_points,
    hash_id,
    embed_batch,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/pipeline.py (helpers only)
# ---------------------------------------------------------------------------
from scripts.ingest.pipeline import (
    _detect_repo_name_from_path,
    detect_language,
    build_information,
    pseudo_backfill_tick,
)

# ---------------------------------------------------------------------------
# Re-exports from ingest/cli.py
# ---------------------------------------------------------------------------
from scripts.ingest.cli import (
    parse_args,
)

# ---------------------------------------------------------------------------
# Additional imports for backward compatibility
# ---------------------------------------------------------------------------
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False

if TYPE_CHECKING:
    from fastembed import TextEmbedding

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # type: ignore

from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.utils import lex_hash_vector_text as _lex_hash_vector_text
from scripts.utils import lex_sparse_vector_text as _lex_sparse_vector_text

try:
    from scripts.ast_analyzer import get_ast_analyzer, chunk_code_semantically
    _AST_ANALYZER_AVAILABLE = True
except ImportError:
    _AST_ANALYZER_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

_TS_WARNED = False


# ---------------------------------------------------------------------------
# Main orchestration functions (kept in façade for test monkeypatching)
# ---------------------------------------------------------------------------
def index_single_file(
    client: QdrantClient,
    model,
    collection: str,
    vector_name: str,
    file_path: Path,
    *,
    dedupe: bool = True,
    skip_unchanged: bool = True,
    pseudo_mode: str = "full",
    trust_cache: bool | None = None,
    repo_name_for_cache: str | None = None,
) -> bool:
    """Index a single file path. Returns True if indexed, False if skipped."""
    try:
        if _should_skip_explicit_file_by_excluder(file_path):
            try:
                delete_points_by_path(client, collection, str(file_path))
            except Exception:
                pass
            print(f"Skipping excluded file: {file_path}")
            return False
    except Exception:
        return False

    _file_lock_ctx = None
    if file_indexing_lock is not None:
        try:
            _file_lock_ctx = file_indexing_lock(str(file_path))
            _file_lock_ctx.__enter__()
        except FileExistsError:
            print(f"[FILE_LOCKED] Skipping {file_path} - another process is indexing it")
            return False
        except Exception:
            pass

    try:
        return _index_single_file_inner(
            client, model, collection, vector_name, file_path,
            dedupe=dedupe, skip_unchanged=skip_unchanged, pseudo_mode=pseudo_mode,
            trust_cache=trust_cache, repo_name_for_cache=repo_name_for_cache,
        )
    finally:
        if _file_lock_ctx is not None:
            try:
                _file_lock_ctx.__exit__(None, None, None)
            except Exception:
                pass


def _index_single_file_inner(
    client: QdrantClient,
    model,
    collection: str,
    vector_name: str,
    file_path: Path,
    *,
    dedupe: bool = True,
    skip_unchanged: bool = True,
    pseudo_mode: str = "full",
    trust_cache: bool | None = None,
    repo_name_for_cache: str | None = None,
) -> bool:
    """Inner implementation of index_single_file (after lock is acquired)."""
    if trust_cache is None:
        try:
            trust_cache = os.environ.get("INDEX_TRUST_CACHE", "").strip().lower() in {
                "1", "true", "yes", "on",
            }
        except Exception:
            trust_cache = False

    fast_fs = _env_truthy(os.environ.get("INDEX_FS_FASTPATH"), False)
    if skip_unchanged and fast_fs and get_cached_file_meta is not None:
        try:
            repo_for_cache = repo_name_for_cache or _detect_repo_name_from_path(file_path)
            meta = get_cached_file_meta(str(file_path), repo_for_cache) or {}
            size = meta.get("size")
            mtime = meta.get("mtime")
            if size is not None and mtime is not None:
                st = file_path.stat()
                if int(getattr(st, "st_size", 0)) == int(size) and int(
                    getattr(st, "st_mtime", 0)
                ) == int(mtime):
                    print(f"Skipping unchanged file (fs-meta): {file_path}")
                    return False
        except Exception:
            pass

    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return False

    language = detect_language(file_path)
    file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    repo_tag = repo_name_for_cache or _detect_repo_name_from_path(file_path)

    repo_id: str | None = None
    repo_rel_path: str | None = None
    if logical_repo_reuse_enabled() and get_workspace_state is not None:
        try:
            ws_root = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
            state = get_workspace_state(ws_root, repo_tag)
            lrid = state.get("logical_repo_id") if isinstance(state, dict) else None
            if isinstance(lrid, str) and lrid:
                repo_id = lrid
            try:
                fp = file_path.resolve()
            except Exception:
                fp = file_path
            try:
                ws_base = Path(os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work").resolve()
                repo_root = ws_base
                if repo_tag:
                    candidate = ws_base / repo_tag
                    if candidate.exists():
                        repo_root = candidate
                rel = fp.relative_to(repo_root)
                repo_rel_path = rel.as_posix()
            except Exception:
                repo_rel_path = None
        except Exception as e:
            print(f"[logical_repo] Failed to derive logical identity for {file_path}: {e}")

    changed_symbols = set()
    if get_cached_symbols and set_cached_symbols:
        cached_symbols = get_cached_symbols(str(file_path))
        if cached_symbols:
            current_symbols = extract_symbols_with_tree_sitter(str(file_path))
            _, changed = compare_symbol_changes(cached_symbols, current_symbols)
            for symbol_data in current_symbols.values():
                symbol_id = f"{symbol_data['type']}_{symbol_data['name']}_{symbol_data['start_line']}"
                if symbol_id in changed:
                    changed_symbols.add(symbol_id)

    if skip_unchanged:
        try:
            if get_cached_file_hash:
                prev_local = get_cached_file_hash(str(file_path), repo_tag)
                if prev_local and file_hash and prev_local == file_hash:
                    if fast_fs and set_cached_file_hash:
                        try:
                            set_cached_file_hash(str(file_path), file_hash, repo_tag)
                        except Exception:
                            pass
                    print(f"Skipping unchanged file (cache): {file_path}")
                    return False
        except Exception:
            pass

        if not trust_cache:
            prev = get_indexed_file_hash(
                client, collection, str(file_path),
                repo_id=repo_id, repo_rel_path=repo_rel_path,
            )
            if prev and prev == file_hash:
                if fast_fs and set_cached_file_hash:
                    try:
                        set_cached_file_hash(str(file_path), file_hash, repo_tag)
                    except Exception:
                        pass
                print(f"Skipping unchanged file: {file_path}")
                return False

    if dedupe:
        delete_points_by_path(client, collection, str(file_path))

    symbols = _extract_symbols(language, text)
    imports, calls = _get_imports_calls(language, text)
    last_mod, churn_count, author_count = _git_metadata(file_path)

    CHUNK_LINES = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    CHUNK_OVERLAP = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)
    use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").lower() in {"1", "true", "yes", "on"}
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").lower() in {"1", "true", "yes", "on"}

    if use_micro:
        try:
            _cap = int(os.environ.get("MAX_MICRO_CHUNKS_PER_FILE", "200") or 200)
            _base_tokens = int(os.environ.get("MICRO_CHUNK_TOKENS", "128") or 128)
            _base_stride = int(os.environ.get("MICRO_CHUNK_STRIDE", "64") or 64)
            chunks = chunk_by_tokens(text, k_tokens=_base_tokens, stride_tokens=_base_stride)
            if _cap > 0 and len(chunks) > _cap:
                _before = len(chunks)
                _scale = (len(chunks) / _cap) * 1.1
                _new_tokens = max(_base_tokens, int(_base_tokens * _scale))
                _new_stride = max(_base_stride, int(_base_stride * _scale))
                chunks = chunk_by_tokens(text, k_tokens=_new_tokens, stride_tokens=_new_stride)
        except Exception:
            chunks = chunk_by_tokens(text)
    elif use_semantic:
        chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
    else:
        chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)

    batch_texts: List[str] = []
    batch_meta: List[Dict] = []
    batch_ids: List[int] = []
    batch_lex: List[list] = []
    batch_lex_text: List[str] = []

    def make_point(pid, dense_vec, lex_vec, payload, lex_text: str = ""):
        if vector_name:
            vecs = {vector_name: dense_vec, LEX_VECTOR_NAME: lex_vec}
            try:
                if os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
                    vecs[MINI_VECTOR_NAME] = project_mini(list(dense_vec), MINI_VEC_DIM)
            except Exception:
                pass
            if LEX_SPARSE_MODE and lex_text:
                sparse_vec = _lex_sparse_vector_text(lex_text)
                if sparse_vec.get("indices"):
                    vecs[LEX_SPARSE_NAME] = models.SparseVector(**sparse_vec)
            return models.PointStruct(id=pid, vector=vecs, payload=payload)
        else:
            return models.PointStruct(id=pid, vector=dense_vec, payload=payload)

    pseudo_batch_concurrency = int(os.environ.get("PSEUDO_BATCH_CONCURRENCY", "1") or 1)
    use_batch_pseudo = pseudo_batch_concurrency > 1 and pseudo_mode == "full"

    chunk_data: list = []
    for ch in chunks:
        info = build_information(
            language, file_path, ch["start"], ch["end"],
            ch["text"].splitlines()[0] if ch["text"] else "",
        )
        kind, sym, sym_path = _choose_symbol_for_chunk(ch["start"], ch["end"], symbols)
        if "kind" in ch and ch.get("kind"):
            kind = ch.get("kind") or kind
        if "symbol" in ch and ch.get("symbol"):
            sym = ch.get("symbol") or sym
        if "symbol_path" in ch and ch.get("symbol_path"):
            sym_path = ch.get("symbol_path") or sym_path
        if not ch.get("kind") and kind:
            ch["kind"] = kind
        if not ch.get("symbol") and sym:
            ch["symbol"] = sym
        if not ch.get("symbol_path") and sym_path:
            ch["symbol_path"] = sym_path

        _cur_path = str(file_path)
        _host_path, _container_path = _compute_host_and_container_paths(_cur_path)

        payload = {
            "document": info,
            "information": info,
            "metadata": {
                "path": str(file_path),
                "path_prefix": str(file_path.parent),
                "ext": str(file_path.suffix).lstrip(".").lower(),
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
                "repo_id": repo_id,
                "repo_rel_path": repo_rel_path,
                "host_path": _host_path,
                "container_path": _container_path,
            },
        }

        needs_pseudo_gen = False
        cached_pseudo, cached_tags = "", []
        if pseudo_mode != "off":
            needs_pseudo_gen, cached_pseudo, cached_tags = should_process_pseudo_for_chunk(
                str(file_path), ch, changed_symbols
            )

        chunk_data.append({
            "chunk": ch,
            "info": info,
            "payload": payload,
            "kind": kind,
            "needs_pseudo": needs_pseudo_gen and pseudo_mode == "full",
            "cached_pseudo": cached_pseudo,
            "cached_tags": cached_tags,
        })

    if use_batch_pseudo:
        pending_indices = [i for i, cd in enumerate(chunk_data) if cd["needs_pseudo"]]
        pending_texts = [chunk_data[i]["chunk"].get("text") or "" for i in pending_indices]

        if pending_texts:
            try:
                from scripts.refrag_glm import generate_pseudo_tags_batch
                batch_results = generate_pseudo_tags_batch(pending_texts, concurrency=pseudo_batch_concurrency)
                for idx, (pseudo, tags) in zip(pending_indices, batch_results):
                    chunk_data[idx]["cached_pseudo"] = pseudo
                    chunk_data[idx]["cached_tags"] = tags
                    chunk_data[idx]["needs_pseudo"] = False
                    if pseudo or tags:
                        ch = chunk_data[idx]["chunk"]
                        symbol_name = ch.get("symbol", "")
                        if symbol_name and set_cached_pseudo:
                            k = ch.get("kind", "unknown")
                            start_line = ch.get("start", 0)
                            symbol_id = f"{k}_{symbol_name}_{start_line}"
                            set_cached_pseudo(str(file_path), symbol_id, pseudo, tags, file_hash)
            except Exception as e:
                print(f"[PSEUDO_BATCH] Batch failed, falling back to sequential: {e}")
                use_batch_pseudo = False

    for cd in chunk_data:
        ch = cd["chunk"]
        payload = cd["payload"]
        pseudo = cd["cached_pseudo"]
        tags = cd["cached_tags"]

        if not use_batch_pseudo and cd["needs_pseudo"]:
            try:
                pseudo, tags = generate_pseudo_tags(ch.get("text") or "")
                if pseudo or tags:
                    symbol_name = ch.get("symbol", "")
                    if symbol_name:
                        kind = ch.get("kind", "unknown")
                        start_line = ch.get("start", 0)
                        symbol_id = f"{kind}_{symbol_name}_{start_line}"
                        if set_cached_pseudo:
                            set_cached_pseudo(str(file_path), symbol_id, pseudo, tags, file_hash)
            except Exception:
                pass

        if pseudo:
            payload["pseudo"] = pseudo
        if tags:
            payload["tags"] = tags
        batch_texts.append(cd["info"])
        batch_meta.append(payload)
        batch_ids.append(hash_id(ch["text"], str(file_path), ch["start"], ch["end"]))
        aug_lex_text = (ch.get("text") or "") + (" " + pseudo if pseudo else "") + (" " + " ".join(tags) if tags else "")
        batch_lex.append(_lex_hash_vector_text(aug_lex_text))
        batch_lex_text.append(aug_lex_text)

    if batch_texts:
        vectors = embed_batch(model, batch_texts)
        for _idx, _m in enumerate(batch_meta):
            try:
                _m["pid_str"] = str(batch_ids[_idx])
            except Exception:
                pass
        points = [
            make_point(i, v, lx, m, lt)
            for i, v, lx, m, lt in zip(batch_ids, vectors, batch_lex, batch_meta, batch_lex_text)
        ]
        upsert_points(client, collection, points)
        try:
            if set_cached_file_hash:
                set_cached_file_hash(str(file_path), file_hash, repo_tag)
        except Exception:
            pass
        return True
    return False


def index_repo(
    root: Path,
    qdrant_url: str,
    api_key: Optional[str],
    collection: Optional[str],
    model_name: str,
    recreate: bool,
    *,
    dedupe: bool = True,
    skip_unchanged: bool = True,
    pseudo_mode: str = "full",
    clear_caches: bool = False,
):
    """Index a repository into Qdrant."""
    fast_fs = _env_truthy(os.environ.get("INDEX_FS_FASTPATH"), False)
    if skip_unchanged and not recreate and fast_fs and get_cached_file_meta is not None:
        try:
            is_multi_repo = bool(is_multi_repo_mode and is_multi_repo_mode())
            root_repo_for_cache = (
                _detect_repo_name_from_path(root)
                if (not is_multi_repo and _detect_repo_name_from_path)
                else None
            )
            all_unchanged = True
            for file_path in iter_files(root):
                per_file_repo_for_cache = (
                    root_repo_for_cache
                    if root_repo_for_cache is not None
                    else (
                        _detect_repo_name_from_path(file_path)
                        if _detect_repo_name_from_path
                        else None
                    )
                )
                meta = get_cached_file_meta(str(file_path), per_file_repo_for_cache) or {}
                size = meta.get("size")
                mtime = meta.get("mtime")
                if size is None or mtime is None:
                    all_unchanged = False
                    break
                st = file_path.stat()
                if int(getattr(st, "st_size", 0)) != int(size) or int(getattr(st, "st_mtime", 0)) != int(mtime):
                    all_unchanged = False
                    break
            if all_unchanged:
                print("[fast_index] No changes detected via fs metadata; skipping model and Qdrant setup")
                return
        except Exception:
            pass

    try:
        from scripts.embedder import get_embedding_model, get_model_dimension
        model = get_embedding_model(model_name)
        dim = get_model_dimension(model_name)
    except ImportError:
        model = TextEmbedding(model_name=model_name)
        dim = len(next(model.embed(["dimension probe"])))

    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key or None,
        timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
    )

    if recreate:
        vector_name = _sanitize_vector_name(model_name)
    else:
        vector_name = None
        try:
            info = client.get_collection(collection)
            cfg = info.config.params.vectors
            if isinstance(cfg, dict) and cfg:
                for name, params in cfg.items():
                    psize = getattr(params, "size", None) or getattr(params, "dim", None)
                    if psize and int(psize) == int(dim):
                        vector_name = name
                        break
                if vector_name is None and LEX_VECTOR_NAME in cfg:
                    for name in cfg.keys():
                        if name != LEX_VECTOR_NAME:
                            vector_name = name
                            break
        except Exception:
            pass
        if vector_name is None:
            vector_name = _sanitize_vector_name(model_name)

    if recreate:
        recreate_collection(client, collection, dim, vector_name)

    try:
        ensure_collection_and_indexes_once(client, collection, dim, vector_name)
    except Exception:
        ensure_collection(client, collection, dim, vector_name)
        ensure_payload_indexes(client, collection)

    is_multi_repo = bool(is_multi_repo_mode and is_multi_repo_mode())
    root_repo_for_cache = (
        _detect_repo_name_from_path(root)
        if (not is_multi_repo and _detect_repo_name_from_path)
        else None
    )

    try:
        files = list(iter_files(root))
        iterator = tqdm(files, desc="Indexing files") if tqdm else files
    except Exception:
        files = list(iter_files(root))
        iterator = files

    for file_path in iterator:
        per_file_repo_for_cache = (
            root_repo_for_cache
            if root_repo_for_cache is not None
            else (
                _detect_repo_name_from_path(file_path)
                if _detect_repo_name_from_path
                else None
            )
        )
        try:
            index_single_file(
                client, model, collection, vector_name, file_path,
                dedupe=dedupe, skip_unchanged=skip_unchanged,
                pseudo_mode=pseudo_mode,
                repo_name_for_cache=per_file_repo_for_cache,
            )
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")


def process_file_with_smart_reindexing(
    file_path,
    text: str,
    language: str,
    client: QdrantClient,
    current_collection: str,
    per_file_repo,
    model,
    vector_name: str | None,
) -> str:
    """Smart, chunk-level reindexing for a single file.

    Rebuilds all points for the file with *accurate* line numbers while:
    - Reusing existing embeddings/lexical vectors for unchanged chunks (by code content), and
    - Re-embedding only for changed chunks.
    """
    try:
        p = Path(str(file_path))
        if _should_skip_explicit_file_by_excluder(p):
            try:
                delete_points_by_path(client, current_collection, str(p))
            except Exception:
                pass
            print(f"[SMART_REINDEX] Skipping excluded file: {file_path}")
            return "skipped"
    except Exception:
        return "skipped"

    print(f"[SMART_REINDEX] Processing {file_path} with chunk-level reindexing")

    try:
        fp = str(file_path)
    except Exception:
        fp = str(file_path)
    try:
        if not isinstance(file_path, Path):
            file_path = Path(fp)
    except Exception:
        file_path = Path(fp)

    file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    symbol_meta = extract_symbols_with_tree_sitter(fp)
    if not symbol_meta:
        print(f"[SMART_REINDEX] No symbols found in {file_path}, falling back to full reindex")
        return "failed"

    cached_symbols = get_cached_symbols(fp) if get_cached_symbols else {}
    unchanged_symbols: list = []
    changed_symbols: list = []
    if cached_symbols and compare_symbol_changes:
        try:
            unchanged_symbols, changed_symbols = compare_symbol_changes(
                cached_symbols, symbol_meta
            )
        except Exception:
            unchanged_symbols = []
            changed_symbols = list(symbol_meta.keys())
    else:
        changed_symbols = list(symbol_meta.keys())
    changed_set = set(changed_symbols)

    if len(changed_symbols) == 0 and cached_symbols:
        print(f"[SMART_REINDEX] {file_path}: 0 changes detected, skipping")
        return "skipped"

    existing_points = []
    try:
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path", match=models.MatchValue(value=fp)
                )
            ]
        )
        next_offset = None
        while True:
            pts, next_offset = client.scroll(
                collection_name=current_collection,
                scroll_filter=filt,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=next_offset,
            )
            if not pts:
                break
            existing_points.extend(pts)
            if next_offset is None:
                break
    except Exception as e:
        print(f"[SMART_REINDEX] Failed to load existing points for {file_path}: {e}")
        existing_points = []

    points_by_code: dict = {}
    try:
        for rec in existing_points:
            payload = rec.payload or {}
            md = payload.get("metadata") or {}
            code_text = md.get("code") or ""
            embed_text = payload.get("information") or payload.get("document") or ""
            kind = md.get("kind") or ""
            sym_name = md.get("symbol") or ""
            start_line = md.get("start_line") or 0
            symbol_id = (
                f"{kind}_{sym_name}_{start_line}"
                if kind and sym_name and start_line
                else ""
            )
            key = (symbol_id, code_text, embed_text) if symbol_id else ("", code_text, embed_text)
            points_by_code.setdefault(key, []).append(rec)
    except Exception:
        points_by_code = {}

    CHUNK_LINES = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    CHUNK_OVERLAP = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)
    use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").lower() in {"1", "true", "yes", "on"}
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").lower() in {"1", "true", "yes", "on"}

    if use_micro:
        chunks = chunk_by_tokens(text)
    elif use_semantic:
        chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
    else:
        chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)
    
    symbol_spans = _extract_symbols(language, text)

    reused_points: list = []
    embed_texts: list = []
    embed_payloads: list = []
    embed_ids: list = []
    embed_lex: list = []
    embed_lex_text: list = []

    imports, calls = _get_imports_calls(language, text)
    last_mod, churn_count, author_count = _git_metadata(file_path)

    pseudo_batch_concurrency = int(os.environ.get("PSEUDO_BATCH_CONCURRENCY", "1") or 1)
    use_batch_pseudo = pseudo_batch_concurrency > 1

    chunk_data_sr: list = []
    for ch in chunks:
        info = build_information(
            language, file_path, ch["start"], ch["end"],
            ch["text"].splitlines()[0] if ch["text"] else "",
        )
        kind, sym, sym_path = _choose_symbol_for_chunk(ch["start"], ch["end"], symbol_spans)
        if "kind" in ch and ch.get("kind"):
            kind = ch.get("kind") or kind
        if "symbol" in ch and ch.get("symbol"):
            sym = ch.get("symbol") or sym
        if "symbol_path" in ch and ch.get("symbol_path"):
            sym_path = ch.get("symbol_path") or sym_path
        if not ch.get("kind") and kind:
            ch["kind"] = kind
        if not ch.get("symbol") and sym:
            ch["symbol"] = sym
        if not ch.get("symbol_path") and sym_path:
            ch["symbol_path"] = sym_path

        _cur_path = str(file_path)
        _host_path, _container_path = _compute_host_and_container_paths(_cur_path)

        payload = {
            "document": info,
            "information": info,
            "metadata": {
                "path": str(file_path),
                "path_prefix": str(file_path.parent),
                "ext": str(file_path.suffix).lstrip(".").lower(),
                "language": language,
                "kind": kind,
                "symbol": sym,
                "symbol_path": sym_path or "",
                "repo": per_file_repo,
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
                "host_path": _host_path,
                "container_path": _container_path,
            },
        }

        needs_pseudo_gen, cached_pseudo, cached_tags = should_process_pseudo_for_chunk(
            fp, ch, changed_set
        )

        chunk_data_sr.append({
            "chunk": ch,
            "info": info,
            "payload": payload,
            "kind": kind,
            "sym": sym,
            "sym_path": sym_path,
            "needs_pseudo": needs_pseudo_gen,
            "cached_pseudo": cached_pseudo,
            "cached_tags": cached_tags,
        })

    if use_batch_pseudo:
        pending_indices = [i for i, cd in enumerate(chunk_data_sr) if cd["needs_pseudo"]]
        pending_texts = [chunk_data_sr[i]["chunk"].get("text") or "" for i in pending_indices]

        if pending_texts:
            try:
                from scripts.refrag_glm import generate_pseudo_tags_batch
                batch_results = generate_pseudo_tags_batch(pending_texts, concurrency=pseudo_batch_concurrency)
                for idx, (pseudo, tags) in zip(pending_indices, batch_results):
                    chunk_data_sr[idx]["cached_pseudo"] = pseudo
                    chunk_data_sr[idx]["cached_tags"] = tags
                    chunk_data_sr[idx]["needs_pseudo"] = False
                    if pseudo or tags:
                        ch = chunk_data_sr[idx]["chunk"]
                        symbol_name = ch.get("symbol", "")
                        if symbol_name and set_cached_pseudo:
                            k = ch.get("kind", "unknown")
                            start_line = ch.get("start", 0)
                            sid = f"{k}_{symbol_name}_{start_line}"
                            set_cached_pseudo(fp, sid, pseudo, tags, file_hash)
            except Exception as e:
                print(f"[PSEUDO_BATCH] Smart reindex batch failed, falling back: {e}")
                use_batch_pseudo = False

    for cd in chunk_data_sr:
        ch = cd["chunk"]
        payload = cd["payload"]
        pseudo = cd["cached_pseudo"]
        tags = cd["cached_tags"]

        if not use_batch_pseudo and cd["needs_pseudo"]:
            try:
                pseudo, tags = generate_pseudo_tags(ch.get("text") or "")
                if pseudo or tags:
                    symbol_name = ch.get("symbol", "")
                    if symbol_name:
                        k = ch.get("kind", "unknown")
                        start_line = ch.get("start", 0)
                        sid = f"{k}_{symbol_name}_{start_line}"
                        if set_cached_pseudo:
                            set_cached_pseudo(fp, sid, pseudo, tags, file_hash)
            except Exception:
                pass

        if pseudo:
            payload["pseudo"] = pseudo
        if tags:
            payload["tags"] = tags

        info = cd["info"]
        kind = cd["kind"]
        sym = cd["sym"]

        code_text = ch.get("text") or ""
        chunk_symbol_id = ""
        if sym and kind:
            chunk_symbol_id = f"{kind}_{sym}_{ch['start']}"

        reuse_key = (chunk_symbol_id, code_text, info)
        fallback_key = ("", code_text, info)
        reused_rec = None
        used_key = None
        bucket = points_by_code.get(reuse_key)
        if bucket is not None:
            used_key = reuse_key
        else:
            bucket = points_by_code.get(fallback_key)
            if bucket is not None:
                used_key = fallback_key
        if bucket:
            try:
                reused_rec = bucket.pop()
                if not bucket:
                    if used_key is not None:
                        points_by_code.pop(used_key, None)
            except Exception:
                reused_rec = None

        if reused_rec is not None:
            try:
                vec = reused_rec.vector
                if vector_name and isinstance(vec, dict) and vector_name not in vec:
                    raise ValueError("reused vector missing dense key")
                aug_lex_text = (code_text or "") + (" " + pseudo if pseudo else "") + (
                    " " + " ".join(tags) if tags else ""
                )
                refreshed_lex = _lex_hash_vector_text(aug_lex_text)
                if vector_name:
                    if isinstance(vec, dict):
                        vec = dict(vec)
                        vec[LEX_VECTOR_NAME] = refreshed_lex
                    else:
                        vecs = {vector_name: vec, LEX_VECTOR_NAME: refreshed_lex}
                        try:
                            if os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
                                vecs[MINI_VECTOR_NAME] = project_mini(list(vec), MINI_VEC_DIM)
                        except Exception:
                            pass
                        vec = vecs
                else:
                    if isinstance(vec, dict):
                        dense = None
                        try:
                            for k, v in vec.items():
                                if k not in {LEX_VECTOR_NAME, MINI_VECTOR_NAME}:
                                    dense = v
                                    break
                        except Exception:
                            dense = None
                        if dense is None:
                            raise ValueError("reused vector has no dense component")
                        vec = dense
                pid = hash_id(code_text, fp, ch["start"], ch["end"])
                reused_points.append(
                    models.PointStruct(id=pid, vector=vec, payload=payload)
                )
                continue
            except Exception:
                pass

        embed_texts.append(info)
        embed_payloads.append(payload)
        embed_ids.append(hash_id(code_text, fp, ch["start"], ch["end"]))
        aug_lex_text = (code_text or "") + (" " + pseudo if pseudo else "") + (" " + " ".join(tags) if tags else "")
        embed_lex.append(_lex_hash_vector_text(aug_lex_text))
        embed_lex_text.append(aug_lex_text)

    new_points: list = []
    if embed_texts:
        vectors = embed_batch(model, embed_texts)
        for pid, v, lx, pl, lt in zip(embed_ids, vectors, embed_lex, embed_payloads, embed_lex_text):
            if vector_name:
                vecs = {vector_name: v, LEX_VECTOR_NAME: lx}
                try:
                    if os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
                        vecs[MINI_VECTOR_NAME] = project_mini(list(v), MINI_VEC_DIM)
                except Exception:
                    pass
                if LEX_SPARSE_MODE and lt:
                    sparse_vec = _lex_sparse_vector_text(lt)
                    if sparse_vec.get("indices"):
                        vecs[LEX_SPARSE_NAME] = models.SparseVector(**sparse_vec)
                new_points.append(models.PointStruct(id=pid, vector=vecs, payload=pl))
            else:
                new_points.append(models.PointStruct(id=pid, vector=v, payload=pl))

    all_points = reused_points + new_points

    try:
        delete_points_by_path(client, current_collection, fp)
    except Exception as e:
        print(f"[SMART_REINDEX] Failed to delete old points for {file_path}: {e}")

    if all_points:
        upsert_points(client, current_collection, all_points)

    try:
        if set_cached_symbols:
            set_cached_symbols(fp, symbol_meta, file_hash)
    except Exception as e:
        print(f"[SMART_REINDEX] Failed to update symbol cache for {file_path}: {e}")
    try:
        if set_cached_file_hash:
            set_cached_file_hash(fp, file_hash, per_file_repo)
    except Exception:
        pass

    print(
        f"[SMART_REINDEX] Completed {file_path}: chunks={len(chunks)}, reused_points={len(reused_points)}, embedded_points={len(new_points)}"
    )
    return "success"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main():
    """Main entry point for the CLI."""
    from scripts.ingest.cli import main as _cli_main
    _cli_main()


# ---------------------------------------------------------------------------
# Public API (__all__)
# ---------------------------------------------------------------------------
__all__ = [
    # Config
    "ROOT_DIR",
    "LEX_VECTOR_NAME",
    "LEX_VECTOR_DIM",
    "MINI_VECTOR_NAME",
    "MINI_VEC_DIM",
    "LEX_SPARSE_NAME",
    "LEX_SPARSE_MODE",
    "CODE_EXTS",
    "EXTENSIONLESS_FILES",
    # Workspace state
    "is_multi_repo_mode",
    "get_collection_name",
    "logical_repo_reuse_enabled",
    "get_cached_symbols",
    "set_cached_symbols",
    "compare_symbol_changes",
    "get_cached_pseudo",
    "set_cached_pseudo",
    "get_cached_file_hash",
    "set_cached_file_hash",
    # Tree-sitter
    "_TS_AVAILABLE",
    "_TS_LANGUAGES",
    "_use_tree_sitter",
    # Vectors
    "project_mini",
    "_lex_hash_vector",
    # Exclusions
    "iter_files",
    "is_indexable_file",
    "_Excluder",
    # Chunking
    "chunk_lines",
    "chunk_semantic",
    "chunk_by_tokens",
    # Symbols
    "_extract_symbols",
    "extract_symbols_with_tree_sitter",
    "_choose_symbol_for_chunk",
    # Pseudo
    "_pseudo_describe_enabled",
    "_smart_symbol_reindexing_enabled",
    "generate_pseudo_tags",
    "should_process_pseudo_for_chunk",
    "should_use_smart_reindexing",
    # Metadata
    "_git_metadata",
    "_get_imports_calls",
    # Qdrant
    "ensure_collection",
    "recreate_collection",
    "ensure_payload_indexes",
    "ensure_collection_and_indexes_once",
    "get_indexed_file_hash",
    "delete_points_by_path",
    "upsert_points",
    "hash_id",
    "embed_batch",
    # Pipeline
    "_detect_repo_name_from_path",
    "detect_language",
    "build_information",
    "index_single_file",
    "index_repo",
    "process_file_with_smart_reindexing",
    "pseudo_backfill_tick",
    # CLI
    "main",
    # Backward compat
    "TextEmbedding",
    "_EMBEDDER_FACTORY",
]


if __name__ == "__main__":
    main()
