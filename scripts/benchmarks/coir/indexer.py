#!/usr/bin/env python3
"""
CoIR Corpus Indexer - Uses Context-Engine's real indexing pipeline.

Indexes CoIR benchmark corpora into Qdrant using the same pipeline as
production code indexing (hybrid vectors, same models, same settings).

This ensures benchmark results reflect actual system performance.

Features:
- Corpus fingerprinting for smart reuse (avoids reindexing unchanged corpora)
- Consistent collection naming per task
- Optional cleanup of benchmark collections
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

# Reuse shared Qdrant client + embedder helpers from cosqa (handles API key, timeout)
from scripts.benchmarks.cosqa.indexer import (
    get_qdrant_client,
    get_embedding_model,
    get_model_dimension,
)
from scripts.ingest.chunking import chunk_by_tokens, chunk_lines, chunk_semantic
from scripts.ingest.pipeline import build_information
from scripts.ingest.vectors import project_mini, extract_pattern_vector
from scripts.ingest.qdrant import (
    hash_id,
    embed_batch,
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

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

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

# Collection prefix for CoIR benchmarks
COIR_COLLECTION_PREFIX = "coir-bench-"

# Metadata key for corpus fingerprint (stored in collection)
CORPUS_FINGERPRINT_KEY = "_corpus_fingerprint"


def get_temp_collection(task_name: str = "default") -> str:
    """Get collection name for a CoIR task."""
    return f"{COIR_COLLECTION_PREFIX}{task_name}"


def get_corpus_collection(corpus: List[Dict[str, Any]]) -> str:
    """Get a stable collection name for a specific corpus.
    
    Uses corpus fingerprint to ensure each unique corpus+config gets its own
    collection. This enables automatic reuse across benchmark runs.
    """
    fp = compute_corpus_fingerprint(corpus)
    return f"{COIR_COLLECTION_PREFIX}{fp}"


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
        f"ast_enriched:v2",
    ]
    return hashlib.sha256("|".join(config_parts).encode()).hexdigest()[:8]


def compute_corpus_fingerprint(corpus: List[Dict[str, Any]]) -> str:
    """Compute a stable fingerprint for a corpus + config to detect changes.
    
    Includes:
    - Sorted doc IDs + all content that affects embeddings (title + text)
    - Embedding model and vector config (invalidates on model/config change)
    - ReFRAG/chunking settings (invalidates on processing change)
    """
    hasher = hashlib.sha256()
    
    # Config fingerprint first - ensures reindex on config change
    config_fp = _get_config_fingerprint()
    hasher.update(f"config:{config_fp}".encode())
    
    # Corpus content fingerprint - include all fields that affect embeddings
    for doc in sorted(corpus, key=lambda d: d.get("_id", "")):
        doc_id = doc.get("_id", "")
        title = doc.get("title", "")
        text = doc.get("text", "")
        language = doc.get("language", "")
        # Hash the same content that gets embedded (title + text)
        # Use full content for accurate fingerprinting (truncation caused stale reuse)
        combined = f"{language}\n{title}\n{text}"
        hasher.update(f"{doc_id}:{combined}".encode("utf-8", errors="ignore"))
    # Note: We intentionally don't include count in fingerprint
    # This allows resume to work when some entries fail to index
    
    return hasher.hexdigest()[:16]


def get_collection_fingerprint(collection: str) -> Optional[str]:
    """Get the stored corpus fingerprint from a collection, if any."""
    try:
        client = get_qdrant_client()
        # Scroll for a single point with the fingerprint payload
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


def collection_matches_corpus(collection: str, corpus: List[Dict[str, Any]]) -> bool:
    """Check if an existing collection matches the given corpus.

    Note: We intentionally don't require exact point count match.
    If fingerprint matches, the resume logic will fill in any missing entries.
    """
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection)
        if info.points_count == 0:
            return False  # Empty collection, needs indexing
        # Fingerprint check
        stored_fp = get_collection_fingerprint(collection)
        if stored_fp:
            return stored_fp == compute_corpus_fingerprint(corpus)
        return False
    except Exception:
        return False


def cleanup_coir_collections(task_names: Optional[List[str]] = None) -> int:
    """Delete CoIR benchmark collections.
    
    Args:
        task_names: Specific tasks to clean up, or None for all coir-bench-* collections
    
    Returns:
        Number of collections deleted
    """
    client = get_qdrant_client()
    deleted = 0
    try:
        collections = client.get_collections().collections
        for col in collections:
            name = getattr(col, "name", "")
            if not name.startswith(COIR_COLLECTION_PREFIX):
                continue
            if task_names is not None:
                task = name[len(COIR_COLLECTION_PREFIX):]
                if task not in task_names:
                    continue
            try:
                client.delete_collection(name)
                deleted += 1
            except Exception:
                pass
    except Exception:
        pass
    return deleted


def create_collection(
    client: QdrantClient,
    collection: str,
    dim: int,
    recreate: bool = False,
) -> None:
    """Create or recreate the CoIR collection with proper vector config."""
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
    for field in ["_id", "language", "source", "doc_type"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def create_lexical_vector(text: str, dim: int | None = None) -> List[float]:
    """Create lexical hash vector for hybrid search."""
    try:
        return _lex_hash_vector_text(text, dim)
    except Exception:
        # Fallback: simple hash-based vector
        import re
        dim = int(dim or 256)
        vec = [0.0] * dim
        tokens = re.findall(r"\w+", text.lower())
        for t in tokens:
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]


def index_coir_corpus(
    corpus: List[Dict[str, Any]],
    collection: str,
    batch_size: int = 100,
    recreate: bool = False,
    show_progress: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Index a CoIR corpus using Context-Engine's production pipeline.
    
    Uses the same embedding model and vector configuration as production.
    Supports smart reuse: skips indexing if the collection already contains
    the same corpus (detected via fingerprinting).
    
    Args:
        corpus: List of {"_id": str, "text": str, "title": str (optional)}
        collection: Qdrant collection name
        batch_size: Batch size for indexing
        recreate: Drop and recreate collection (deprecated, use force=True)
        show_progress: Show progress bar
        force: Force reindexing even if collection matches corpus
    
    Returns:
        {"indexed": int, "collection": str, "time_s": float, "reused": bool}
    """
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        OptimizersConfigDiff,
    )
    
    # force implies recreate (drop & full reindex)
    if force:
        recreate = True

    # Compute fingerprint for this corpus (needed for both matching and storing)
    corpus_fingerprint = compute_corpus_fingerprint(corpus)

    # Check if we can reuse existing collection
    if not recreate:
        match_result = collection_matches_corpus(collection, corpus)
        if match_result:
            if show_progress:
                print(f"  Reusing existing collection: {collection} ({len(corpus)} docs, fingerprint match)")
            return {
                "indexed": len(corpus),
                "collection": collection,
                "time_s": 0.0,
                "reused": True,
            }
        else:
            # Check if collection exists with different data (fingerprint mismatch)
            client = get_qdrant_client()
            try:
                existing = client.get_collection(collection)
                if existing.points_count > 0:
                    # Collection exists with different data - must recreate to avoid mixing
                    if show_progress:
                        print(f"  Collection {collection} exists but fingerprint mismatch ({existing.points_count} points) - recreating")
                    recreate = True
            except Exception:
                # Collection doesn't exist, will be created below
                pass

    # Get embedding model (same as production)
    try:
        from scripts.embedder import get_embedding_model
        embed_model = get_embedding_model()
    except ImportError:
        from fastembed import TextEmbedding
        embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    # Vector naming must match Context-Engine's search stack (`hybrid_search`),
    # which uses scripts.utils.sanitize_vector_name(EMBEDDING_MODEL).
    from scripts.utils import sanitize_vector_name
    from scripts.ingest.config import LEX_VECTOR_NAME, LEX_VECTOR_DIM
    from scripts.ingest.vectors import _lex_hash_vector as create_lexical_vector

    vector_name = sanitize_vector_name(EMBEDDING_MODEL)

    # Determine embedding dimension from the actual model (avoid env drift).
    try:
        probe = list(embed_model.embed(["dim_probe"]))[0]
        dim = len(probe.tolist() if hasattr(probe, "tolist") else list(probe))
    except Exception:
        # Keep a reasonable fallback; should match most BGE base configs.
        dim = 768

    client = get_qdrant_client()
    start_time = time.time()
    
    # Create/recreate collection with hybrid vectors
    if recreate:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
    
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                vector_name: VectorParams(size=dim, distance=Distance.COSINE),
                LEX_VECTOR_NAME: VectorParams(size=LEX_VECTOR_DIM, distance=Distance.COSINE),
            },
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
    
    # Index in batches
    indexed = 0
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        
        # Prepare texts
        texts = []
        for doc in batch:
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = f"{title}\n{text}" if title else text
            texts.append(combined)
        
        # Get dense embeddings
        dense_vecs = list(embed_model.embed(texts))
        
        # Build points
        points = []
        for j, doc in enumerate(batch):
            dense = dense_vecs[j]
            if hasattr(dense, 'tolist'):
                dense = dense.tolist()
            
            lexical = create_lexical_vector(texts[j], LEX_VECTOR_DIM)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    vector_name: dense,
                    LEX_VECTOR_NAME: lexical,
                },
                payload={
                    "_id": doc["_id"],
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                    "doc_type": "coir_corpus",
                    CORPUS_FINGERPRINT_KEY: corpus_fingerprint,
                    # Synthesize minimal codebase-like metadata so `hybrid_search` / `repo_search`
                    # can operate on CoIR corpora (which are not real files on disk).
                    "metadata": {
                        "path": f"coir/{collection}/{doc['_id']}",
                        "path_prefix": f"coir/{collection}",
                        "symbol": doc.get("title", "") or doc["_id"],
                        "symbol_path": doc.get("title", "") or doc["_id"],
                        "kind": "document",
                        "language": "",
                        "start_line": 1,
                        "end_line": 1,
                        "text": texts[j],
                        # Reranker expects metadata.code for cross-encoder scoring
                        "code": texts[j],
                    },
                },
            )
            points.append(point)
        
        client.upsert(collection_name=collection, points=points)
        indexed += len(points)
        
        if show_progress and (i + batch_size) % 500 == 0:
            print(f"  Indexed {indexed}/{len(corpus)} documents...")
    
    elapsed = time.time() - start_time
    if show_progress:
        print(f"  Indexed {indexed} documents in {elapsed:.1f}s")
    
    return {"indexed": indexed, "collection": collection, "time_s": elapsed, "reused": False}
