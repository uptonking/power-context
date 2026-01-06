#!/usr/bin/env python3
"""
CoSQA Corpus Indexer - Thin adapter over shared benchmark indexing core.

Transforms CoSQA dataset entries into generic BenchmarkDoc format and delegates
to scripts/benchmarks/core_indexer.py for actual indexing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.benchmarks.core_indexer import (
    BenchmarkDoc,
    index_benchmark_corpus,
    compute_corpus_fingerprint as _compute_corpus_fingerprint,
    get_collection_fingerprint as _get_collection_fingerprint,
    collection_matches_corpus as _collection_matches_corpus,
)
from scripts.benchmarks.qdrant_utils import (
    get_qdrant_client,
    get_embedding_model,
)

# Default configuration
DEFAULT_COLLECTION = "cosqa-corpus"
DEFAULT_BATCH_SIZE = 100


def cosqa_entry_to_doc(entry: Dict[str, Any]) -> BenchmarkDoc:
    """Convert CoSQA entry to BenchmarkDoc."""
    code_id = entry.get("code_id", "")
    text = entry.get("text", "")
    docstring = entry.get("docstring", "")
    func_name = entry.get("func_name", "")

    # Combine docstring + code for embedding (matches original indexer)
    full_text = f"{docstring}\n{text}".strip() if docstring else text

    return BenchmarkDoc(
        doc_id=code_id,
        text=full_text,
        language="python",
        metadata={
            "code_id": code_id,
            "func_name": func_name,
            "docstring": docstring,
            "source": "cosqa",
        },
    )


def compute_corpus_fingerprint(corpus_entries: List[Dict[str, Any]]) -> str:
    """Compute fingerprint for CoSQA corpus."""
    docs = [cosqa_entry_to_doc(e) for e in corpus_entries]
    return _compute_corpus_fingerprint(docs)


def get_collection_fingerprint(collection: str) -> str | None:
    """Get stored fingerprint from collection."""
    client = get_qdrant_client()
    return _get_collection_fingerprint(client, collection)


def collection_matches_corpus(collection: str, corpus_entries: List[Dict[str, Any]]) -> bool:
    """Check if collection matches corpus fingerprint."""
    client = get_qdrant_client()
    docs = [cosqa_entry_to_doc(e) for e in corpus_entries]
    return _collection_matches_corpus(client, collection, docs)


def index_corpus(
    corpus_entries: List[Dict[str, Any]],
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    recreate: bool = False,
    resume: bool = True,
    progress_callback: callable | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Index CoSQA corpus into Qdrant using shared benchmark indexer.

    Args:
        corpus_entries: List of corpus entries from dataset.get_corpus_for_indexing()
        collection: Qdrant collection name
        batch_size: Batch size for upsert operations
        recreate: Whether to recreate the collection
        resume: Ignored (kept for API compatibility)
        progress_callback: Ignored (kept for API compatibility)
        force: Force reindexing even if collection matches corpus

    Returns:
        Stats dict with indexed count, time, errors, reused
    """
    # CoSQA snippets are atomic units - disable chunking so 1 point = 1 snippet.
    # This ensures points_count ≈ corpus_size for proper rerank/candidate scaling.
    os.environ.setdefault("INDEX_SEMANTIC_CHUNKS", "0")
    os.environ.setdefault("INDEX_CHUNK_LINES", "10000")
    os.environ.setdefault("INDEX_CHUNK_OVERLAP", "0")

    # Convert CoSQA entries to generic BenchmarkDoc format
    docs = [cosqa_entry_to_doc(e) for e in corpus_entries]

    # Get client and model
    client = get_qdrant_client()
    model = get_embedding_model()

    # Delegate to shared benchmark indexer
    if force:
        recreate = True

    result = index_benchmark_corpus(
        docs=docs,
        client=client,
        model=model,
        collection=collection,
        batch_size=batch_size,
        recreate=recreate,
        skip_if_exists=not force,
    )

    # Convert result format to match old API
    return {
        "indexed": result.get("indexed_count", 0),
        "skipped": result.get("skipped_count", 0),
        "errors": 0,
        "reused": result.get("reused", False),
    }


def verify_collection(collection: str = DEFAULT_COLLECTION) -> Dict[str, Any]:
    """Verify collection exists and return stats."""
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
