#!/usr/bin/env python3
from __future__ import annotations

# Canonical vector-name sanitizer used by both ingest and search paths
# Keeps mapping consistent across the codebase

def sanitize_vector_name(model_name: str) -> str:
    name = (model_name or "").strip().lower()
    # Common fastembed alias mapping for MiniLM
    if name in (
        "sentence-transformers/all-minilm-l6-v2",
        "sentence-transformers/all-minilm-l-6-v2",
        "sentence-transformers/all-minilm-l6-v2",
    ):
        return "fast-all-minilm-l6-v2"
    # Common fastembed alias mapping for BGE base
    if "bge-base-en-v1.5" in name:
        return "fast-bge-base-en-v1.5"
    # Fallback: compact name
    return name.replace("/", "-").replace("_", "-")[:64]

