"""Centralized embedder factory with Qwen3 feature flag support.

This module provides a unified interface for embedding model initialization,
supporting both the default BGE-base model and the optional Qwen3-Embedding
model via feature flags.

Environment Variables:
    EMBEDDING_MODEL: Model name (default: BAAI/bge-base-en-v1.5)
    QWEN3_EMBEDDING_ENABLED: Enable Qwen3 model registration (0/1)
    QWEN3_QUERY_INSTRUCTION: Add instruction prefix to queries (0/1)
    QWEN3_INSTRUCTION_TEXT: Custom instruction prefix text
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

# Default model configuration
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QWEN3_MODEL = "electroglyph/Qwen3-Embedding-0.6B-onnx-uint8"
QWEN3_DIM = 1024

# Feature flags
QWEN3_ENABLED = (
    str(os.environ.get("QWEN3_EMBEDDING_ENABLED", "0")).strip().lower()
    in {"1", "true", "yes", "on"}
)
QWEN3_QUERY_INSTRUCTION = (
    str(os.environ.get("QWEN3_QUERY_INSTRUCTION", "1")).strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_INSTRUCTION = (
    "Instruct: Given a code search query, retrieve relevant code snippets\nQuery:"
)

# Model cache and locks
_EMBED_MODEL_CACHE: Dict[str, Any] = {}
_EMBED_MODEL_LOCKS: Dict[str, threading.Lock] = {}
_QWEN3_REGISTERED = False
_QWEN3_REGISTER_LOCK = threading.Lock()


def _register_qwen3_model() -> None:
    """Register Qwen3 ONNX model with FastEmbed (one-time, thread-safe)."""
    global _QWEN3_REGISTERED
    if _QWEN3_REGISTERED:
        return

    with _QWEN3_REGISTER_LOCK:
        if _QWEN3_REGISTERED:
            return
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType

            TextEmbedding.add_custom_model(
                model=QWEN3_MODEL,
                pooling=PoolingType.DISABLED,
                normalization=False,
                sources=ModelSource(hf=QWEN3_MODEL),
                dim=QWEN3_DIM,
                model_file="dynamic_uint8.onnx",
            )
            _QWEN3_REGISTERED = True
        except Exception:
            # Registration failed - model may already exist or fastembed issue
            pass


def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """Get or create a cached embedding model instance.

    Args:
        model_name: Model name override. If None, uses EMBEDDING_MODEL env var.

    Returns:
        TextEmbedding instance (cached per model name).
    """
    from fastembed import TextEmbedding

    if model_name is None:
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)

    # Register Qwen3 if enabled and requested
    if QWEN3_ENABLED and "qwen3" in model_name.lower():
        _register_qwen3_model()

    # Check cache first (fast path)
    cached = _EMBED_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    # Double-checked locking for thread safety
    lock = _EMBED_MODEL_LOCKS.setdefault(model_name, threading.Lock())
    with lock:
        cached = _EMBED_MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        model = TextEmbedding(model_name=model_name)
        # Warmup with common code patterns
        try:
            _ = list(model.embed(["function", "class", "import", "def", "const"]))
        except Exception:
            pass

        _EMBED_MODEL_CACHE[model_name] = model
        return model


def is_qwen3_model(model_name: Optional[str] = None) -> bool:
    """Check if the given or configured model is Qwen3."""
    if model_name is None:
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    return "qwen3" in model_name.lower()


def get_query_instruction() -> str:
    """Get the query instruction prefix for Qwen3 models."""
    return os.environ.get("QWEN3_INSTRUCTION_TEXT", DEFAULT_INSTRUCTION)


def prefix_query(query: str, model_name: Optional[str] = None) -> str:
    """Add instruction prefix to query if using Qwen3 with instructions enabled.

    Args:
        query: The search query text.
        model_name: Model name override. If None, uses EMBEDDING_MODEL env var.

    Returns:
        Query with instruction prefix (if applicable) or original query.
    """
    if not QWEN3_QUERY_INSTRUCTION:
        return query
    if not is_qwen3_model(model_name):
        return query
    instruction = get_query_instruction()
    return f"{instruction} {query}"


def prefix_queries(queries: List[str], model_name: Optional[str] = None) -> List[str]:
    """Add instruction prefix to multiple queries if using Qwen3.

    Args:
        queries: List of search query texts.
        model_name: Model name override. If None, uses EMBEDDING_MODEL env var.

    Returns:
        List of queries with instruction prefixes (if applicable).
    """
    if not QWEN3_QUERY_INSTRUCTION:
        return queries
    if not is_qwen3_model(model_name):
        return queries
    instruction = get_query_instruction()
    return [f"{instruction} {q}" for q in queries]


def get_model_dimension(model_name: Optional[str] = None) -> int:
    """Get the embedding dimension for the specified model.

    Args:
        model_name: Model name override. If None, uses EMBEDDING_MODEL env var.

    Returns:
        Embedding dimension for the model.
    """
    if model_name is None:
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)

    # Qwen3 models: 1024 dimensions
    if is_qwen3_model(model_name):
        return QWEN3_DIM

    # Known model dimensions (case-insensitive matching)
    model_lower = model_name.lower()

    # MiniLM models: 384 dimensions
    if "minilm" in model_lower or "all-minilm" in model_lower:
        return 384

    # BGE-small: 384 dimensions
    if "bge-small" in model_lower:
        return 384

    # BGE-large: 1024 dimensions
    if "bge-large" in model_lower:
        return 1024

    # E5 models
    if "e5-small" in model_lower:
        return 384
    if "e5-large" in model_lower:
        return 1024
    if "e5-base" in model_lower:
        return 768

    # Default: BGE-base and similar 768-dimension models
    return 768

