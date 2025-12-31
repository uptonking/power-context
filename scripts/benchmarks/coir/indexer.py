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
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Reuse shared Qdrant client helper from cosqa (handles API key, timeout)
from scripts.benchmarks.cosqa.indexer import get_qdrant_client

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

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
    - ReFRAG/micro-chunk settings
    """
    from scripts.ingest.config import LEX_VECTOR_DIM
    
    config_parts = [
        f"model:{EMBEDDING_MODEL}",
        f"lex_dim:{LEX_VECTOR_DIM}",
        f"refrag:{os.environ.get('REFRAG_ENABLED', 'false')}",
        f"micro_budget:{os.environ.get('MICRO_BUDGET_TOKENS', '0')}",
        f"chunk_size:{os.environ.get('CHUNK_SIZE', '512')}",
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
        # Hash the same content that gets embedded (title + text)
        # Use full content for accurate fingerprinting (truncation caused stale reuse)
        combined = f"{title}\n{text}"
        hasher.update(f"{doc_id}:{combined}".encode("utf-8", errors="ignore"))
    hasher.update(f"count:{len(corpus)}".encode())
    
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
    """Check if an existing collection matches the given corpus."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection)
        # Quick check: point count must match
        if info.points_count != len(corpus):
            return False
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
    
    # Check if we can reuse existing collection
    if not recreate:
        if collection_matches_corpus(collection, corpus):
            if show_progress:
                print(f"  Reusing existing collection: {collection} ({len(corpus)} docs, fingerprint match)")
            return {
                "indexed": len(corpus),
                "collection": collection,
                "time_s": 0.0,
                "reused": True,
            }
    
    # Compute fingerprint for this corpus
    corpus_fingerprint = compute_corpus_fingerprint(corpus)
    
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

