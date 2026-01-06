#!/usr/bin/env python3
"""
Shared benchmark indexing core for CoSQA, CoIR, and other code retrieval benchmarks.

Provides common indexing logic to avoid duplication across benchmark-specific indexers.
Uses Context-Engine's production embedding pipeline (chunking, vectors, symbols).

Key features:
- Generic document format (BenchmarkDoc)
- Config fingerprinting for smart collection reuse
- Batch processing with progress tracking
- AST-aware symbol extraction
- Multi-vector support (dense, lexical, mini, pattern, sparse)
"""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

# Import production pipeline components
from scripts.ingest.chunking import chunk_by_tokens, chunk_lines, chunk_semantic
from scripts.ingest.pipeline import build_information, _select_dense_text
from scripts.ingest.vectors import project_mini, extract_pattern_vector
from scripts.ingest.qdrant import (
    hash_id,
    embed_batch,
    get_collection_vector_names,
    PATTERN_VECTOR_NAME,
    PATTERN_VECTOR_DIM,
    upsert_points as _upsert_points_with_retry,
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

# Optional AST/symbol extraction
try:
    from scripts.ingest.symbols import _extract_symbols, _choose_symbol_for_chunk
    from scripts.ingest.metadata import _get_imports_calls
    _AST_AVAILABLE = True
except ImportError:
    _AST_AVAILABLE = False
    def _extract_symbols(_lang, _text):
        return []
    def _choose_symbol_for_chunk(_start, _end, _symbols):
        return "", "", ""
    def _get_imports_calls(_lang, _text):
        return [], []


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CORPUS_FINGERPRINT_KEY = "_corpus_fingerprint"


@dataclass
class BenchmarkDoc:
    """Generic document for benchmark indexing."""
    doc_id: str  # Unique identifier (code_id, _id, etc.)
    text: str  # Code or text content
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra fields (title, docstring, etc.)


def get_config_fingerprint() -> str:
    """Get fingerprint of current embedding/vector configuration.
    
    Includes all settings that would require reindexing if changed.
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
        f"dense_mode:{os.environ.get('INDEX_DENSE_MODE', 'code+info')}",
        f"ast_enriched:v3",
    ]
    return hashlib.sha256("|".join(config_parts).encode()).hexdigest()[:8]


def compute_corpus_fingerprint(docs: List[BenchmarkDoc]) -> str:
    """Compute stable fingerprint for corpus + config."""
    hasher = hashlib.sha256()
    
    # Config first
    config_fp = get_config_fingerprint()
    hasher.update(f"config:{config_fp}".encode())
    
    # Corpus content
    for doc in sorted(docs, key=lambda d: d.doc_id):
        # Hash doc_id + text + all metadata that affects embeddings
        content = f"{doc.doc_id}:{doc.text}"
        for k, v in sorted(doc.metadata.items()):
            content += f"|{k}:{v}"
        hasher.update(content.encode("utf-8", errors="ignore"))
    
    return hasher.hexdigest()[:16]


def get_collection_fingerprint(client: QdrantClient, collection: str) -> Optional[str]:
    """Retrieve stored corpus fingerprint from collection metadata."""
    try:
        result = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=CORPUS_FINGERPRINT_KEY))]
            ),
            limit=1,
        )
        if result[0]:
            return result[0][0].payload.get("fingerprint")
    except Exception:
        pass
    return None


def collection_matches_corpus(
    client: QdrantClient,
    collection: str,
    docs: List[BenchmarkDoc],
) -> bool:
    """Check if collection exists and matches corpus fingerprint."""
    try:
        info = client.get_collection(collection)
        if info.points_count == 0:
            return False
        stored_fp = get_collection_fingerprint(client, collection)
        if stored_fp:
            return stored_fp == compute_corpus_fingerprint(docs)
        return False
    except Exception:
        return False


def create_collection(
    client: QdrantClient,
    collection: str,
    dim: int,
    recreate: bool = False,
) -> None:
    """Create or recreate collection with proper vector config."""
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

    # Create with named vectors
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

    # Create payload indexes
    for field in ["doc_id", "language", "source"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass


def generate_point_id(doc_id: str) -> str:
    """Generate stable Qdrant point ID from doc_id."""
    return hashlib.md5(doc_id.encode()).hexdigest()


def create_lexical_vector(text: str, dim: int | None = None) -> List[float]:
    """Create lexical hash vector for hybrid search."""
    try:
        return _lex_hash_vector_text(text, dim or LEX_VECTOR_DIM)
    except Exception:
        # Fallback to zero vector on error
        return [0.0] * (dim or LEX_VECTOR_DIM)


def index_benchmark_corpus(
    docs: List[BenchmarkDoc],
    client: QdrantClient,
    model: Any,  # TextEmbedding
    collection: str,
    *,
    batch_size: int = 100,
    max_workers: int = 4,
    recreate: bool = False,
    skip_if_exists: bool = True,
) -> Dict[str, Any]:
    """Index a benchmark corpus into Qdrant.

    Args:
        docs: List of BenchmarkDoc to index
        client: Qdrant client
        model: Embedding model (fastembed TextEmbedding)
        collection: Collection name
        batch_size: Points per batch
        max_workers: Parallel workers for processing
        recreate: Force recreate collection
        skip_if_exists: Skip if collection matches corpus fingerprint

    Returns:
        Dict with stats: indexed_count, skipped_count, duration_sec
    """
    from scripts.utils import sanitize_vector_name

    start_time = time.time()

    # Check if we can skip indexing
    if skip_if_exists and not recreate:
        if collection_matches_corpus(client, collection, docs):
            print(f"Collection {collection} matches corpus fingerprint, skipping indexing")
            return {
                "indexed_count": 0,
                "skipped_count": len(docs),
                "duration_sec": 0.0,
                "reused": True,
            }

    # Get model dimension
    try:
        dim = model.model.get_sentence_embedding_dimension()
    except Exception:
        dim = 768  # Default BGE dimension

    # Create collection
    create_collection(client, collection, dim, recreate=recreate)

    # Get vector names
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    vector_name = sanitize_vector_name(model_name)
    available_vectors = get_collection_vector_names(client, collection)

    # Phase 1: Prepare all chunk metadata (CPU-bound, can parallelize)
    print(f"Indexing {len(docs)} documents into {collection}...")
    print(f"  Phase 1: Preparing chunks (workers={max_workers})...")

    @dataclass
    class ChunkMeta:
        doc_id: str
        chunk_idx: int
        chunk_text: str
        dense_text: str
        payload: Dict[str, Any]
        language: str

    chunk_metas: List[ChunkMeta] = []

    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1") == "1"
    chunk_lines_val = int(os.environ.get("INDEX_CHUNK_LINES", "120"))
    chunk_overlap = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20"))

    def prepare_doc_chunks(doc: BenchmarkDoc) -> List[ChunkMeta]:
        """Prepare chunk metadata for a single document."""
        results = []
        try:
            symbols = []
            if _AST_AVAILABLE:
                try:
                    symbols = _extract_symbols(doc.language, doc.text)
                except Exception:
                    pass

            if use_semantic:
                try:
                    chunks = chunk_semantic(doc.text, doc.language, chunk_lines_val, chunk_overlap)
                except Exception:
                    chunks = chunk_lines(doc.text, chunk_lines_val, chunk_overlap)
            else:
                chunks = chunk_lines(doc.text, chunk_lines_val, chunk_overlap)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")
                start_line = chunk.get("start", 1)
                end_line = chunk.get("end", 1)
                if not chunk_text.strip():
                    continue

                symbol_name = chunk.get("symbol", "")
                symbol_kind = chunk.get("kind", "")
                if not symbol_name and symbols:
                    symbol_kind, symbol_name, _ = _choose_symbol_for_chunk(start_line, end_line, symbols)

                path = doc.metadata.get("path", f"bench/{doc.doc_id}")
                first_line = chunk_text.split("\n")[0] if chunk_text else ""
                info = build_information(doc.language, Path(path), start_line, end_line, first_line)

                pseudo = doc.metadata.get("docstring", "") or doc.metadata.get("title", "")
                tags = [symbol_name] if symbol_name else None
                dense_text = _select_dense_text(
                    info=info,
                    code_text=chunk_text,
                    pseudo=pseudo,
                    tags=tags,
                )

                synthetic_path = doc.metadata.get("path", f"bench/{doc.doc_id}.py")
                payload = {
                    "doc_id": doc.doc_id,
                    "text": chunk_text,
                    "language": doc.language,
                    "start_line": start_line + 1,
                    "end_line": end_line,
                    "symbol": symbol_name,
                    "symbol_kind": symbol_kind,
                    **doc.metadata,
                    "metadata": {
                        "path": synthetic_path,
                        "path_prefix": str(Path(synthetic_path).parent),
                        "language": doc.language,
                        "kind": symbol_kind or "document",
                        "symbol": symbol_name or doc.doc_id,
                        "symbol_path": symbol_name or doc.doc_id,
                        "start_line": start_line + 1,
                        "end_line": end_line,
                        "text": chunk_text,
                        "code": chunk_text,
                    },
                }

                results.append(ChunkMeta(
                    doc_id=doc.doc_id,
                    chunk_idx=chunk_idx,
                    chunk_text=chunk_text,
                    dense_text=dense_text,
                    payload=payload,
                    language=doc.language,
                ))
        except Exception as e:
            print(f"  Warning: Failed to prepare doc {doc.doc_id}: {e}")
        return results

    # Parallel chunk preparation
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(prepare_doc_chunks, doc): doc for doc in docs}
            for future in as_completed(futures):
                chunk_metas.extend(future.result())
    else:
        for doc in docs:
            chunk_metas.extend(prepare_doc_chunks(doc))

    print(f"  Prepared {len(chunk_metas)} chunks from {len(docs)} documents")

    # Phase 2: Batch embedding (GPU/CPU intensive, batch for efficiency)
    print(f"  Phase 2: Batch embedding {len(chunk_metas)} chunks...")
    embed_batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "64"))

    dense_texts = [cm.dense_text for cm in chunk_metas]
    all_dense_vecs: List[List[float]] = []

    for i in range(0, len(dense_texts), embed_batch_size):
        batch_texts = dense_texts[i:i + embed_batch_size]
        batch_vecs = list(model.embed(batch_texts))
        all_dense_vecs.extend([v.tolist() for v in batch_vecs])
        if (i + embed_batch_size) % 500 == 0:
            print(f"    Embedded {min(i + embed_batch_size, len(dense_texts))}/{len(dense_texts)} chunks...")

    # Phase 3: Build points and upsert
    print(f"  Phase 3: Building and upserting points...")
    points = []
    indexed_count = 0

    for idx, cm in enumerate(chunk_metas):
        dense_vec = all_dense_vecs[idx]
        lex_vec = create_lexical_vector(cm.chunk_text)

        vectors_dict = {
            vector_name: dense_vec,
            LEX_VECTOR_NAME: lex_vec,
        }

        if MINI_VECTOR_NAME in available_vectors:
            try:
                mini_vec = project_mini(dense_vec)
                vectors_dict[MINI_VECTOR_NAME] = mini_vec
            except Exception:
                pass

        if PATTERN_VECTOR_NAME in available_vectors:
            try:
                pattern_vec = extract_pattern_vector(cm.chunk_text, cm.language)
                if pattern_vec:
                    vectors_dict[PATTERN_VECTOR_NAME] = pattern_vec
            except Exception:
                pass

        sparse_dict = None
        if LEX_SPARSE_MODE and LEX_SPARSE_NAME in (available_vectors.get("sparse") or set()):
            try:
                sparse_dict = {LEX_SPARSE_NAME: _lex_sparse_vector_text(cm.chunk_text)}
            except Exception:
                pass

        point_id = generate_point_id(f"{cm.doc_id}:{cm.chunk_idx}")
        point = models.PointStruct(
            id=point_id,
            vector=vectors_dict,
            payload=cm.payload,
        )
        if sparse_dict:
            point.vector.update(sparse_dict)  # type: ignore

        points.append(point)

        if len(points) >= batch_size:
            _upsert_points_with_retry(client, collection, points)
            indexed_count += len(points)
            points = []

    # Final batch
    if points:
        _upsert_points_with_retry(client, collection, points)
        indexed_count += len(points)

    # Store fingerprint
    fingerprint = compute_corpus_fingerprint(docs)
    fp_point = models.PointStruct(
        id=generate_point_id(CORPUS_FINGERPRINT_KEY),
        vector={vector_name: [0.0] * dim, LEX_VECTOR_NAME: [0.0] * LEX_VECTOR_DIM},
        payload={"doc_id": CORPUS_FINGERPRINT_KEY, "fingerprint": fingerprint},
    )
    _upsert_points_with_retry(client, collection, [fp_point])

    duration = time.time() - start_time
    print(f"Indexed {indexed_count} chunks from {len(docs)} documents in {duration:.1f}s")

    return {
        "indexed_count": indexed_count,
        "skipped_count": 0,
        "duration_sec": duration,
        "reused": False,
    }

