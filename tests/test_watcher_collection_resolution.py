import importlib
import types
import os
import pytest

pytestmark = pytest.mark.unit

def test_main_resolves_collection_from_state(monkeypatch, tmp_path):
    # Env setup: placeholder collection name at startup
    monkeypatch.setenv("WATCH_ROOT", str(tmp_path))
    monkeypatch.setenv("COLLECTION_NAME", "my-collection")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")

    # Stub workspace_state.get_collection_name to return a resolved name
    ws = importlib.import_module("scripts.workspace_state")
    monkeypatch.setattr(ws, "get_collection_name", lambda p: "repo-abc123", raising=True)

    wi = importlib.import_module("scripts.watch_index")
    # Reload to re-read env defaults (COLLECTION) in module globals
    wi = importlib.reload(wi)

    # Fake QdrantClient: force get_collection to raise so code chooses sanitized vector name path
    class FakeQdrant:
        def __init__(self, *a, **k):
            pass
        def get_collection(self, *a, **k):
            raise RuntimeError("not available in unit test")
    monkeypatch.setattr(wi, "QdrantClient", FakeQdrant, raising=True)

    # Fake TextEmbedding: just yield a fixed-dim vector on probe
    class FakeTE:
        def __init__(self, model_name: str):
            self.model_name = model_name
        def embed(self, texts):
            # Next() will take the first yielded item and len() is the dim
            yield [0.0] * 32
    monkeypatch.setattr(wi, "TextEmbedding", FakeTE, raising=True)

    # No-op ensure calls
    idx = importlib.import_module("scripts.ingest_code")
    monkeypatch.setattr(idx, "ensure_collection", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(idx, "ensure_payload_indexes", lambda *a, **k: None, raising=True)

    # Fake Observer to avoid starting real threads
    class FakeObserver:
        def schedule(self, *a, **k):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def join(self):
            pass
    monkeypatch.setattr(wi, "Observer", FakeObserver, raising=True)

    # Make the main loop exit immediately by raising KeyboardInterrupt on sleep
    def _raise_kb(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr(wi.time, "sleep", _raise_kb, raising=True)

    # Precondition: module-level COLLECTION should reflect placeholder at import time
    assert wi.COLLECTION == os.environ.get("COLLECTION_NAME") == "my-collection"

    # Run main(); it should resolve COLLECTION via get_collection_name before any state writes
    wi.main()

    # Postcondition: global COLLECTION mutated to resolved name, not the placeholder
    assert wi.COLLECTION == "repo-abc123"

