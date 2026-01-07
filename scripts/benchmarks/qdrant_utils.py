#!/usr/bin/env python3
"""
Shared Qdrant + embedder helpers for benchmark indexers.

Benchmarks should not depend on each other (e.g., CoIR importing CoSQA) just to
reuse small utilities. Keep these helpers in a neutral module.
"""
from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client with standard configuration."""
    url = os.environ.get("QDRANT_URL") or "http://localhost:6333"
    api_key = os.environ.get("QDRANT_API_KEY")
    timeout = int(os.environ.get("QDRANT_TIMEOUT") or "60")
    return QdrantClient(url=url, api_key=api_key or None, timeout=timeout)


def get_embedding_model() -> Any:
    """Get embedding model using standard Context-Engine factory."""
    try:
        from scripts.embedder import get_embedding_model as _get_model
        return _get_model()
    except ImportError:
        from fastembed import TextEmbedding

        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        return TextEmbedding(model_name=model_name)


def get_model_dimension(model: Any) -> int:
    """Get embedding dimension from model."""
    try:
        from scripts.embedder import get_model_dimension as _get_dim

        model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        return _get_dim(model_name)
    except ImportError:
        # Probe dimension
        vec = list(model.embed(["test"]))[0]
        return len(vec)

