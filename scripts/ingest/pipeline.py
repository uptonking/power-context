#!/usr/bin/env python3
"""
ingest/pipeline.py - Core indexing pipeline.

This module provides the main indexing functions: index_single_file, index_repo,
process_file_with_smart_reindexing, and related orchestration logic.
"""
from __future__ import annotations

import os
import sys
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from qdrant_client import QdrantClient, models

from scripts.ingest.config import (
    ROOT_DIR,
    CODE_EXTS,
    EXTENSIONLESS_FILES,
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
    MINI_VECTOR_NAME,
    MINI_VEC_DIM,
    _env_truthy,
    is_multi_repo_mode,
    get_collection_name,
    logical_repo_reuse_enabled,
    get_cached_file_hash,
    set_cached_file_hash,
    get_cached_symbols,
    set_cached_symbols,
    get_cached_pseudo,
    set_cached_pseudo,
    get_cached_file_meta,
    get_workspace_state,
    compare_symbol_changes,
    file_indexing_lock,
)
from scripts.ingest.exclusions import (
    iter_files,
    _should_skip_explicit_file_by_excluder,
)
from scripts.ingest.chunking import chunk_lines, chunk_semantic, chunk_by_tokens
from scripts.ingest.symbols import (
    _extract_symbols,
    _choose_symbol_for_chunk,
    extract_symbols_with_tree_sitter,
)
from scripts.ingest.pseudo import (
    generate_pseudo_tags,
    should_process_pseudo_for_chunk,
)
from scripts.ingest.metadata import (
    _git_metadata,
    _get_imports_calls,
    _compute_host_and_container_paths,
)
from scripts.ingest.vectors import project_mini
from scripts.ingest.qdrant import (
    ensure_collection,
    ensure_collection_and_indexes_once,
    ensure_payload_indexes,
    recreate_collection,
    get_indexed_file_hash,
    delete_points_by_path,
    upsert_points,
    hash_id,
    embed_batch,
)

# Import utility functions
from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.utils import lex_hash_vector_text as _lex_hash_vector_text
from scripts.utils import lex_sparse_vector_text as _lex_sparse_vector_text

if TYPE_CHECKING:
    from fastembed import TextEmbedding


def _detect_repo_name_from_path(path: Path) -> str:
    """Wrapper function to use workspace_state repository detection."""
    try:
        from scripts.workspace_state import _extract_repo_name_from_path as _ws_detect
        return _ws_detect(str(path))
    except ImportError:
        return path.name if path.is_dir() else path.parent.name


def detect_language(path: Path) -> str:
    """Detect language from file extension or name pattern."""
    ext_lang = CODE_EXTS.get(path.suffix.lower())
    if ext_lang:
        return ext_lang
    fname_lower = path.name.lower()
    if fname_lower in EXTENSIONLESS_FILES:
        return EXTENSIONLESS_FILES[fname_lower]
    if fname_lower.startswith("dockerfile"):
        return "dockerfile"
    return "unknown"


def build_information(
    language: str, path: Path, start: int, end: int, first_line: str
) -> str:
    """Build the information/document string for a chunk."""
    first_line = (first_line or "").strip()
    if len(first_line) > 160:
        first_line = first_line[:160] + "…"
    return f"{language} code from {path} lines {start}-{end}. {first_line}"


def index_single_file(
    client: QdrantClient,
    model: "TextEmbedding",
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
    model: "TextEmbedding",
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
        ws_path = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
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
                try:
                    print(
                        f"[ingest] micro-chunks resized path={file_path} count={_before}->{len(chunks)} "
                        f"tokens={_base_tokens}->{_new_tokens} stride={_base_stride}->{_new_stride}"
                    )
                except Exception:
                    pass
        except Exception:
            chunks = chunk_by_tokens(text)
    elif use_semantic:
        chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
    else:
        chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)

    batch_texts: List[str] = []
    batch_meta: List[Dict] = []
    batch_ids: List[int] = []
    batch_lex: List[list[float]] = []
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

    chunk_data: list[dict] = []
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
            ws = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
            if set_cached_file_hash:
                file_repo_tag = repo_tag
                set_cached_file_hash(str(file_path), file_hash, file_repo_tag)
        except Exception:
            pass
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
    pseudo_mode: str = "full",
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
                try:
                    print("[fast_index] No changes detected via fs metadata; skipping model and Qdrant setup")
                except Exception:
                    pass
                return
        except Exception:
            pass

    try:
        from scripts.embedder import get_embedding_model, get_model_dimension
        model = get_embedding_model(model_name)
        dim = get_model_dimension(model_name)
    except ImportError:
        from fastembed import TextEmbedding
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
        from tqdm import tqdm
        files = list(iter_files(root))
        iterator = tqdm(files, desc="Indexing files")
    except ImportError:
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

    repo_id: str | None = None
    repo_rel_path: str | None = None
    try:
        repo_tag = per_file_repo
        if logical_repo_reuse_enabled() and get_workspace_state is not None:
            ws_root = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
            state = get_workspace_state(ws_root, repo_tag)
            lrid = state.get("logical_repo_id") if isinstance(state, dict) else None
            if isinstance(lrid, str) and lrid.strip():
                repo_id = lrid.strip()
            try:
                ws_base = Path(ws_root).resolve()
                repo_root = ws_base
                if repo_tag:
                    candidate = ws_base / str(repo_tag)
                    if candidate.exists():
                        repo_root = candidate
                rel = file_path.resolve().relative_to(repo_root)
                repo_rel_path = rel.as_posix()
            except Exception:
                repo_rel_path = None
    except Exception as e:
        print(f"[SMART_REINDEX] Failed to derive logical repo identity for {file_path}: {e}")

    symbol_meta = extract_symbols_with_tree_sitter(fp)
    if not symbol_meta:
        print(f"[SMART_REINDEX] No symbols found in {file_path}, falling back to full reindex")
        return "failed"

    cached_symbols = get_cached_symbols(fp) if get_cached_symbols else {}
    unchanged_symbols: list[str] = []
    changed_symbols: list[str] = []
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

    points_by_code: dict[tuple[str, str, str], list] = {}
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

    reused_points: list[models.PointStruct] = []
    embed_texts: list[str] = []
    embed_payloads: list[dict] = []
    embed_ids: list[int] = []
    embed_lex: list[list[float]] = []
    embed_lex_text: list[str] = []

    imports, calls = _get_imports_calls(language, text)
    last_mod, churn_count, author_count = _git_metadata(file_path)

    pseudo_batch_concurrency = int(os.environ.get("PSEUDO_BATCH_CONCURRENCY", "1") or 1)
    use_batch_pseudo = pseudo_batch_concurrency > 1

    chunk_data_sr: list[dict] = []
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
                "repo_id": repo_id,
                "repo_rel_path": repo_rel_path,
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
                        if LEX_SPARSE_MODE and aug_lex_text:
                            try:
                                sparse_vec = _lex_sparse_vector_text(aug_lex_text)
                                if sparse_vec.get("indices"):
                                    vec[LEX_SPARSE_NAME] = models.SparseVector(**sparse_vec)
                            except Exception:
                                pass
                    else:
                        vecs = {vector_name: vec, LEX_VECTOR_NAME: refreshed_lex}
                        try:
                            if os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
                                vecs[MINI_VECTOR_NAME] = project_mini(list(vec), MINI_VEC_DIM)
                        except Exception:
                            pass
                        if LEX_SPARSE_MODE and aug_lex_text:
                            try:
                                sparse_vec = _lex_sparse_vector_text(aug_lex_text)
                                if sparse_vec.get("indices"):
                                    vecs[LEX_SPARSE_NAME] = models.SparseVector(**sparse_vec)
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

    new_points: list[models.PointStruct] = []
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


# Import pseudo_backfill_tick for backward compatibility
def pseudo_backfill_tick(
    client: QdrantClient,
    collection: str,
    repo_name: str | None = None,
    *,
    max_points: int = 256,
    dim: int | None = None,
    vector_name: str | None = None,
) -> int:
    """Best-effort pseudo/tag backfill for a collection."""
    from scripts.ingest.qdrant import ensure_collection_and_indexes_once
    
    if not collection or max_points <= 0:
        return 0

    try:
        from qdrant_client import models as _models
    except Exception:
        return 0

    must_conditions: list[Any] = []
    if repo_name:
        try:
            must_conditions.append(
                _models.FieldCondition(
                    key="metadata.repo",
                    match=_models.MatchValue(value=repo_name),
                )
            )
        except Exception:
            pass

    flt = None
    try:
        null_cond = getattr(_models, "IsNullCondition", None)
        empty_cond = getattr(_models, "IsEmptyCondition", None)
        if null_cond is not None:
            should_conditions = []
            try:
                should_conditions.append(null_cond(is_null="pseudo"))
            except Exception:
                pass
            try:
                should_conditions.append(null_cond(is_null="tags"))
            except Exception:
                pass
            if empty_cond is not None:
                try:
                    should_conditions.append(empty_cond(is_empty="tags"))
                except Exception:
                    pass
            flt = _models.Filter(
                must=must_conditions or None,
                should=should_conditions or None,
            )
        else:
            flt = _models.Filter(must=must_conditions or None)
    except Exception:
        flt = None

    processed = 0
    debug_enabled = (os.environ.get("PSEUDO_BACKFILL_DEBUG") or "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    next_offset = None

    def _maybe_ensure_collection() -> bool:
        if not dim or not vector_name:
            return False
        try:
            ensure_collection_and_indexes_once(client, collection, int(dim), vector_name)
            return True
        except Exception:
            return False

    while processed < max_points:
        batch_limit = max(1, min(64, max_points - processed))
        try:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=batch_limit,
                with_payload=True,
                with_vectors=True,
                offset=next_offset,
            )
        except Exception:
            if _maybe_ensure_collection():
                try:
                    points, next_offset = client.scroll(
                        collection_name=collection,
                        scroll_filter=flt,
                        limit=batch_limit,
                        with_payload=True,
                        with_vectors=True,
                        offset=next_offset,
                    )
                except Exception:
                    break
            else:
                break

        if not points:
            break

        new_points: list[Any] = []
        for rec in points:
            try:
                payload = rec.payload or {}
                md = payload.get("metadata") or {}
                code = md.get("code") or ""
                if not code:
                    continue

                pseudo = payload.get("pseudo") or ""
                tags_val = payload.get("tags") or []
                tags: list[str] = list(tags_val) if isinstance(tags_val, list) else []

                if not pseudo and not tags:
                    try:
                        pseudo, tags = generate_pseudo_tags(code)
                    except Exception:
                        pseudo, tags = "", []

                if not pseudo and not tags:
                    continue

                payload["pseudo"] = pseudo
                payload["tags"] = tags

                aug_text = f"{code} {pseudo} {' '.join(tags)}".strip()
                lex_vec = _lex_hash_vector_text(aug_text)

                vec = rec.vector
                if isinstance(vec, dict):
                    vecs = dict(vec)
                    vecs[LEX_VECTOR_NAME] = lex_vec
                    new_vec = vecs
                else:
                    new_vec = vec

                if LEX_SPARSE_MODE and aug_text and isinstance(new_vec, dict):
                    sparse_vec = _lex_sparse_vector_text(aug_text)
                    if sparse_vec.get("indices"):
                        new_vec[LEX_SPARSE_NAME] = models.SparseVector(**sparse_vec)

                new_points.append(
                    models.PointStruct(
                        id=rec.id,
                        vector=new_vec,
                        payload=payload,
                    )
                )
                processed += 1
            except Exception:
                continue

        if new_points:
            try:
                upsert_points(client, collection, new_points)
            except Exception:
                if _maybe_ensure_collection():
                    try:
                        upsert_points(client, collection, new_points)
                    except Exception:
                        break
                else:
                    break

        if next_offset is None:
            break

    return processed
