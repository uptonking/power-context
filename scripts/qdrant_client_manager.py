#!/usr/bin/env python3
"""
Qdrant client lifecycle management to prevent socket leaks.
Provides a singleton client that can be safely reused across the application.
"""
import os
import threading
from typing import Optional
from qdrant_client import QdrantClient


_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()


def get_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    force_new: bool = False
) -> QdrantClient:
    """
    Get or create a singleton Qdrant client.
    
    Args:
        url: Qdrant URL (defaults to QDRANT_URL env var)
        api_key: API key (defaults to QDRANT_API_KEY env var)
        force_new: Create a new client instead of reusing singleton
    
    Returns:
        QdrantClient instance
    
    Note: The singleton client is thread-safe for read operations.
    For write-heavy operations, consider using force_new=True or
    call close_qdrant_client() when done with a session.
    """
    global _client
    
    if force_new:
        url = url or os.environ.get("QDRANT_URL", "http://qdrant:6333")
        api_key = api_key or os.environ.get("QDRANT_API_KEY")
        return QdrantClient(url=url, api_key=api_key if api_key else None)
    
    with _client_lock:
        if _client is None:
            url = url or os.environ.get("QDRANT_URL", "http://qdrant:6333")
            api_key = api_key or os.environ.get("QDRANT_API_KEY")
            _client = QdrantClient(url=url, api_key=api_key if api_key else None)
        return _client


def close_qdrant_client():
    """
    Close the singleton Qdrant client.
    Should be called at application shutdown or when switching configurations.
    """
    global _client
    
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None


def reset_qdrant_client(url: Optional[str] = None, api_key: Optional[str] = None):
    """
    Reset the singleton client with new configuration.
    Useful when switching between different Qdrant instances.
    """
    close_qdrant_client()
    return get_qdrant_client(url=url, api_key=api_key)
