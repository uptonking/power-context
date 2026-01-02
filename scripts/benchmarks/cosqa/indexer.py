#!/usr/bin/env python3
"""
CoSQA Corpus Indexer for Context-Engine Benchmarks.

Indexes the CoSQA code corpus into a dedicated Qdrant collection using
our existing embedding pipeline (BGE + lexical vectors).

Features:
- **AST-aware processing**: Uses tree-sitter/AST symbol extraction like main pipeline
- **Semantic chunking**: Uses chunk_semantic for code-aware chunking
- Progress tracking with resume capability
- Uses standard Context-Engine embedding pipeline
- Stores metadata for result mapping back to original IDs
- Config-aware fingerprinting for smart collection reuse
- Automatic invalidation on model/config changes
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from qdrant_client import QdrantClient, models

from scripts.ingest.chunking import chunk_by_tokens, chunk_lines, chunk_semantic
from scripts.ingest.pipeline import build_information
from scripts.ingest.vectors import project_mini, extract_pattern_vector
from scripts.ingest.qdrant import (
    hash_id,
    get_collection_vector_names,
    PATTERN_VECTOR_NAME,
    PATTERN_VECTOR_DIM,
)
from scripts.utils import (
    lex_hash_vector_text as _lex_hash_vector_text,
    lex_sparse_vector_text as _lex_sparse_vector_text,
)
from scripts.ingest.config import (
    LEX_VECTOR_NAME,
    LEX_VECTOR_DIM,
    MINI_VECTOR_NAME,
    MINI_VEC_DIM,
    LEX_SPARSE_NAME,
    LEX_SPARSE_MODE,
)

# Import Context-Engine's AST/symbol extraction (optional)
try:
    from scripts.ingest.symbols import _extract_symbols, _choose_symbol_for_chunk
    from scripts.ingest.metadata import _get_imports_calls
    _AST_AVAILABLE = True
except ImportError:
    _AST_AVAILABLE = False
    def _extract_symbols(_lang, _text):  # noqa: unused args for fallback
        return []
    def _choose_symbol_for_chunk(_start, _end, _symbols):  # noqa: unused args for fallback
        return "", "", ""
    def _get_imports_calls(_lang, _text):  # noqa: unused args for fallback
        return [], []

# Default configuration
DEFAULT_COLLECTION = "cosqa-corpus"
DEFAULT_BATCH_SIZE = 256  # Larger batches for faster embedding throughput
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

# Metadata key for corpus fingerprint (stored in collection)
CORPUS_FINGERPRINT_KEY = "_corpus_fingerprint"


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with standard configuration."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    timeout = int(os.environ.get("QDRANT_TIMEOUT", "60"))
    return QdrantClient(url=url, api_key=api_key or None, timeout=timeout)


def get_embedding_model():
    """Get embedding model using standard Context-Engine factory."""
    try:
        from scripts.embedder import get_embedding_model as _get_model
        return _get_model()
    except ImportError:
        from fastembed import TextEmbedding
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        return TextEmbedding(model_name=model_name)


def get_model_dimension(model) -> int:
    """Get embedding dimension from model."""
    try:
        from scripts.embedder import get_model_dimension as _get_dim
        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        return _get_dim(model_name)
    except ImportError:
        # Probe dimension
        vec = list(model.embed(["test"]))[0]
        return len(vec)


def _get_config_fingerprint() -> str:
    """Get fingerprint of current embedding/vector configuration.

    Includes all settings that would require reindexing if changed:
    - Embedding model
    - Lexical vector dimensions
    - ReFRAG/mini vector settings
    - Chunking settings (semantic vs micro)
    - Sparse/pattern vector toggles
    - AST enrichment (v2 = enriched with symbols/imports/calls)
    """
    from scripts.ingest.config import LEX_VECTOR_DIM, MINI_VEC_DIM

    config_parts = [
        f"model:{EMBEDDING_MODEL}",
        f"lex_dim:{LEX_VECTOR_DIM}",
        f"refrag_mode:{os.environ.get('REFRAG_MODE', '0')}",
        f"mini_dim:{MINI_VEC_DIM}",
        f"mini_seed:{os.environ.get('MINI_VEC_SEED', '1337')}",
        f"lex_sparse:{os.environ.get('LEX_SPARSE_MODE', '0')}",
        f"pattern_vectors:{os.environ.get('PATTERN_VECTORS', '0')}",
        f"index_micro:{os.environ.get('INDEX_MICRO_CHUNKS', '0')}",
        f"micro_tokens:{os.environ.get('MICRO_CHUNK_TOKENS', '16')}",
        f"micro_stride:{os.environ.get('MICRO_CHUNK_STRIDE', '')}",
        f"semantic_chunks:{os.environ.get('INDEX_SEMANTIC_CHUNKS', '1')}",
        f"chunk_lines:{os.environ.get('INDEX_CHUNK_LINES', '120')}",
        f"chunk_overlap:{os.environ.get('INDEX_CHUNK_OVERLAP', '20')}",
        f"use_tree_sitter:{os.environ.get('USE_TREE_SITTER', '1')}",
        f"enhanced_ast:{os.environ.get('INDEX_USE_ENHANCED_AST', '1')}",
        f"ast_enriched:v2",  # Triggers reindex when AST enrichment changes
    ]
    return hashlib.sha256("|".join(config_parts).encode()).hexdigest()[:8]


def compute_corpus_fingerprint(corpus_entries: List[Dict[str, Any]]) -> str:
    """Compute a stable fingerprint for a corpus + config to detect changes.
    
    Includes:
    - Sorted code_ids + all content that affects embeddings (text, docstring, func_name)
    - Embedding model and vector config (invalidates on model/config change)
    - ReFRAG/mini vector settings (invalidates on processing change)
    """
    hasher = hashlib.sha256()
    
    # Config fingerprint first - ensures reindex on config change
    config_fp = _get_config_fingerprint()
    hasher.update(f"config:{config_fp}".encode())
    
    # Corpus content fingerprint - include all fields that affect embeddings
    for entry in sorted(corpus_entries, key=lambda e: e.get("code_id", "")):
        code_id = entry.get("code_id", "")
        text = entry.get("text", "")
        docstring = entry.get("docstring", "")
        func_name = entry.get("func_name", "")
        # Hash the same content that gets embedded (docstring + text, func_name for lexical)
        # Use full content for accurate fingerprinting (truncation caused stale reuse)
        combined = f"{func_name}\n{docstring}\n{text}"
        hasher.update(f"{code_id}:{combined}".encode("utf-8", errors="ignore"))
    # Note: We intentionally don't include count in fingerprint
    # This allows resume to work when some entries fail to index
    
    return hasher.hexdigest()[:16]


def get_collection_fingerprint(collection: str) -> Optional[str]:
    """Get the stored corpus fingerprint from a collection, if any."""
    try:
        client = get_qdrant_client()
        points, _ = client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=[CORPUS_FINGERPRINT_KEY],
        )
        if points and points[0].payload:
            return points[0].payload.get(CORPUS_FINGERPRINT_KEY)
    except Exception:
        pass
    return None


def collection_matches_corpus(collection: str, corpus_entries: List[Dict[str, Any]]) -> bool:
    """Check if an existing collection matches the given corpus.
    
    Note: We intentionally don't require exact point count match.
    If fingerprint matches, the resume logic will fill in any missing entries.
    """
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection)
        if info.points_count == 0:
            return False  # Empty collection, needs indexing
        # Fingerprint check (content-based, not count-based)
        stored_fp = get_collection_fingerprint(collection)
        if stored_fp:
            return stored_fp == compute_corpus_fingerprint(corpus_entries)
        return False
    except Exception:
        return False


def create_collection(
    client: QdrantClient,
    collection: str,
    dim: int,
    recreate: bool = False,
) -> None:
    """Create or recreate the CoSQA collection with proper vector config."""
    from scripts.ingest.config import (
        LEX_VECTOR_NAME,
        LEX_VECTOR_DIM,
        MINI_VECTOR_NAME,
        MINI_VEC_DIM,
        LEX_SPARSE_NAME,
        LEX_SPARSE_MODE,
    )
    from scripts.ingest.qdrant import PATTERN_VECTOR_NAME, PATTERN_VECTOR_DIM

    # Use sanitized model name for vector name (matches hybrid_search)
    from scripts.utils import sanitize_vector_name
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vector_name = sanitize_vector_name(model_name)

    if recreate:
        try:
            client.delete_collection(collection)
            print(f"Deleted existing collection: {collection}")
        except Exception:
            pass

    # Check if exists
    try:
        info = client.get_collection(collection)
        print(f"Collection {collection} exists with {info.points_count} points")
        return
    except Exception:
        pass

    # Create with named vectors (dense + lexical)
    vectors_config = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(size=LEX_VECTOR_DIM, distance=models.Distance.COSINE),
    }
    refrag_on = os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    if refrag_on:
        vectors_config[MINI_VECTOR_NAME] = models.VectorParams(
            size=MINI_VEC_DIM, distance=models.Distance.COSINE
        )
    pattern_on = os.environ.get("PATTERN_VECTORS", "").strip().lower() in {"1", "true", "yes", "on"}
    if pattern_on:
        vectors_config[PATTERN_VECTOR_NAME] = models.VectorParams(
            size=PATTERN_VECTOR_DIM, distance=models.Distance.COSINE
        )

    sparse_cfg = None
    if LEX_SPARSE_MODE:
        sparse_cfg = {
            LEX_SPARSE_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(full_scan_threshold=5000)
            )
        }

    client.create_collection(
        collection_name=collection,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )
    print(f"Created collection: {collection} (dim={dim})")

    # Create payload indexes for filtering
    for field in ["code_id", "language", "source"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def generate_point_id(code_id: str) -> str:
    """Generate stable Qdrant point ID from code_id."""
    return hashlib.md5(code_id.encode()).hexdigest()


def create_lexical_vector(text: str, dim: int | None = None) -> List[float]:
    """Create lexical hash vector for hybrid search."""
    try:
        return _lex_hash_vector_text(text, dim)
    except Exception:
        # Fallback: simple hash-based vector
        import re
        dim = int(dim or 256)
        vec = [0.0] * dim
        tokens = re.findall(r'\w+', text.lower())
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
        norm = (sum(v*v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]


def index_corpus(
    corpus_entries: List[Dict[str, Any]],
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = False,
    resume: bool = True,
    progress_callback: Optional[callable] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Index CoSQA corpus into Qdrant.

    Supports smart reuse: skips indexing if the collection already contains
    the same corpus with matching config (detected via fingerprinting).

    Args:
        corpus_entries: List of corpus entries from dataset.get_corpus_for_indexing()
        collection: Qdrant collection name
        batch_size: Batch size for upsert operations
        recreate: Whether to recreate the collection (deprecated, use force=True)
        resume: Whether to skip already-indexed entries (queries Qdrant)
        progress_callback: Optional callback(indexed, total) for progress updates
        force: Force reindexing even if collection matches corpus

    Returns:
        Stats dict with indexed count, time, errors, reused
    """
    from scripts.ingest.config import (
        LEX_VECTOR_NAME,
        LEX_VECTOR_DIM,
        MINI_VECTOR_NAME,
        MINI_VEC_DIM,
    )
    from scripts.ingest.vectors import project_mini

    # force implies recreate (drop & full reindex)
    if force:
        recreate = True

    # Compute fingerprint for this corpus upfront
    corpus_fingerprint = compute_corpus_fingerprint(corpus_entries)
    client = get_qdrant_client()

    # Check if we can reuse existing collection (fingerprint match)
    if not recreate:
        if collection_matches_corpus(collection, corpus_entries):
            print(f"Reusing existing collection: {collection} ({len(corpus_entries)} entries, fingerprint match)")
            return {
                "indexed": len(corpus_entries),
                "skipped": 0,
                "errors": 0,
                "reused": True,
            }
        
        # Check if collection exists with different fingerprint
        # Instead of recreating, we'll try to resume and fill in missing entries
        stored_fp = get_collection_fingerprint(collection)
        if stored_fp is not None and stored_fp != corpus_fingerprint:
            print(f"Fingerprint mismatch for {collection} (stored={stored_fp}, current={corpus_fingerprint})")
            print(f"  → Will resume and fill in missing entries (not recreating)")
            # Don't set recreate=True - let resume logic handle it
        elif stored_fp is None:
            # Legacy collection without fingerprint - check if it has points
            try:
                info = client.get_collection(collection)
                if info.points_count > 0:
                    print(f"Collection {collection} has no fingerprint but {info.points_count} points; recreating for AST-enriched index")
                    recreate = True
            except Exception:
                pass  # Collection doesn't exist, will be created fresh

    model = get_embedding_model()
    dim = get_model_dimension(model)
    # Use sanitized model name for vector name (matches hybrid_search/repo_search)
    from scripts.utils import sanitize_vector_name
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vector_name = sanitize_vector_name(model_name)

    # Create collection (recreate=True will drop existing)
    create_collection(client, collection, dim, recreate=recreate)

    refrag_on = os.environ.get("REFRAG_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    use_mini = False
    if refrag_on:
        try:
            info = client.get_collection(collection)
            vcfg = info.config.params.vectors
            if isinstance(vcfg, dict) and MINI_VECTOR_NAME in vcfg:
                use_mini = True
            else:
                print(
                    f"[COLLECTION_WARNING] Collection {collection} lacks mini vector "
                    f"'{MINI_VECTOR_NAME}'. ReFRAG vectors will be skipped."
                )
        except Exception:
            pass

    # Build set of already-indexed code_ids by querying Qdrant (skip if recreating)
    indexed_set = set()
    if resume and not recreate:
        try:
            # Scroll through collection to get all code_ids
            next_offset = None
            page = 0
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=next_offset,
                    with_payload=["code_id"],
                )
                if not points:
                    break
                for pt in points:
                    if pt.payload and pt.payload.get("code_id"):
                        indexed_set.add(pt.payload["code_id"])
                page += 1
                if next_offset is None:
                    break
            if indexed_set:
                print(f"Found {len(indexed_set)} already-indexed entries in Qdrant (scanned {page} pages)")
        except Exception as e:
            print(f"Warning: Could not query existing entries: {e}")

    # Filter out already indexed
    to_index = [e for e in corpus_entries if e["code_id"] not in indexed_set]

    if not to_index:
        print(f"All {len(corpus_entries)} entries already indexed")
        return {"indexed": 0, "skipped": len(corpus_entries), "errors": 0, "reused": True}

    print(f"Indexing {len(to_index)} entries (skipping {len(indexed_set)} already indexed)")

    stats = {"indexed": 0, "skipped": len(indexed_set), "errors": 0, "batches": 0}
    start_time = time.time()
    total = len(to_index)
    last_pct = 0

    # Split into batches
    batches = [to_index[i:i + batch_size] for i in range(0, len(to_index), batch_size)]

    # Parallel upsert workers (embedding is sequential, upsert is parallel)
    N_WORKERS = 10

    def upsert_points(points_list, worker_client):
        """Upsert pre-built points to Qdrant."""
        worker_client.upsert(collection_name=collection, points=points_list)
        return len(points_list)

    n_chunks = (len(batches) + N_WORKERS - 1) // N_WORKERS
    print(f"  Processing {len(batches)} batches in {n_chunks} chunks of {N_WORKERS}...")

    # Process in chunks: embed sequentially (not thread-safe), upsert in parallel
    for chunk_idx, chunk_start in enumerate(range(0, len(batches), N_WORKERS)):
        chunk = batches[chunk_start:chunk_start + N_WORKERS]
        chunk_points = []

        # Embed each batch sequentially (fastembed not thread-safe)
        for batch_idx, batch in enumerate(chunk):
            print(f"  Chunk {chunk_idx+1}/{n_chunks} batch {batch_idx+1}/{len(chunk)}: embedding {len(batch)} items...", end=" ", flush=True)

            # Extract AST symbols and enrich text for better code understanding
            texts = []
            symbol_data = []
            for e in batch:
                code = e.get('text', '')
                docstring = e.get('docstring', '')
                func_name = e.get('func_name', '')

                # Extract symbols using Context-Engine's AST pipeline (with error handling)
                symbols, imports, calls = [], [], []
                if _AST_AVAILABLE and code:
                    try:
                        symbols = _extract_symbols("python", code) or []
                        imports, calls = _get_imports_calls("python", code) or ([], [])
                    except Exception:
                        pass  # Fallback to empty for malformed code

                # Build enriched text for embedding:
                # - Function/class names (from AST)
                # - Imports (dependencies)
                # - Docstring (semantic intent)
                # - Code (implementation)
                symbol_names = [s.get('name', '') for s in symbols if s.get('name')]
                import_names = [getattr(imp, 'module', str(imp)) if hasattr(imp, 'module') else str(imp) for imp in imports[:10]]
                call_names = [getattr(c, 'name', str(c)) if hasattr(c, 'name') else str(c) for c in calls[:10]]

                # Create enriched text: symbols + imports + docstring + code
                enriched_parts = []
                if symbol_names:
                    enriched_parts.append(f"# Defines: {', '.join(symbol_names[:10])}")
                if import_names:
                    enriched_parts.append(f"# Imports: {', '.join(import_names)}")
                if call_names:
                    enriched_parts.append(f"# Calls: {', '.join(call_names)}")
                if docstring:
                    enriched_parts.append(f'"""{docstring}"""')
                enriched_parts.append(code)

                enriched_text = '\n'.join(enriched_parts)
                texts.append(enriched_text)
                symbol_data.append({
                    "symbols": symbol_names[:20],
                    "imports": import_names,
                    "calls": call_names,
                })

            embeddings = list(model.embed(texts))
            print("done", flush=True)

            points = []
            for i, entry in enumerate(batch):
                code_id = entry["code_id"]
                point_id = generate_point_id(code_id)
                emb = embeddings[i]
                dense_vec = emb.tolist() if hasattr(emb, 'tolist') else list(emb)

                # Lexical vector also includes symbol names for better keyword matching
                lex_parts = [
                    entry.get('func_name', ''),
                    entry.get('docstring', ''),
                    entry.get('text', ''),
                    ' '.join(symbol_data[i].get('symbols', [])),
                    ' '.join(symbol_data[i].get('imports', [])),
                    ' '.join(symbol_data[i].get('calls', [])),
                ]
                lex_text = ' '.join(lex_parts)
                lex_vec = create_lexical_vector(lex_text, LEX_VECTOR_DIM)

                # Store symbol metadata in payload
                entry_with_fp = {
                    **entry,
                    CORPUS_FINGERPRINT_KEY: corpus_fingerprint,
                    "symbols": symbol_data[i].get("symbols", []),
                    "imports": symbol_data[i].get("imports", []),
                    "calls": symbol_data[i].get("calls", []),
                }
                vecs = {vector_name: dense_vec, LEX_VECTOR_NAME: lex_vec}
                if use_mini:
                    try:
                        vecs[MINI_VECTOR_NAME] = project_mini(dense_vec, MINI_VEC_DIM)
                    except Exception:
                        pass
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vecs,
                    payload=entry_with_fp,
                ))
            chunk_points.append(points)

        # Upsert in parallel (Qdrant client is thread-safe with separate instances)
        with ThreadPoolExecutor(max_workers=len(chunk_points)) as executor:
            worker_clients = [get_qdrant_client() for _ in chunk_points]
            futures = {
                executor.submit(upsert_points, pts, wc): len(pts)
                for pts, wc in zip(chunk_points, worker_clients)
            }
            for future in as_completed(futures):
                batch_size = futures[future]
                try:
                    count = future.result()
                    stats["indexed"] += count
                    stats["batches"] += 1
                except Exception as e:
                    print(f"  Batch error (size={batch_size}): {e}")
                    stats["errors"] += batch_size

        # Progress after each chunk (always log)
        elapsed = time.time() - start_time
        rate = stats["indexed"] / elapsed if elapsed > 0 else 0
        pct = int(100 * stats["indexed"] / total) if total else 100
        print(f"  Chunk {chunk_idx+1}/{n_chunks} done: {stats['indexed']}/{total} ({pct}%, {rate:.1f}/s)")
        last_pct = pct

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["rate_per_second"] = round(stats["indexed"] / elapsed, 2) if elapsed > 0 else 0
    stats["reused"] = False

    print(f"\nIndexing complete: {stats['indexed']} entries in {elapsed:.1f}s")
    return stats


def verify_collection(collection: str = DEFAULT_COLLECTION) -> Dict[str, Any]:
    """Verify the indexed collection."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection)
        return {
            "exists": True,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}
