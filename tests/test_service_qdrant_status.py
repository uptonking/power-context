import types
import importlib
import pytest

srv = importlib.import_module("scripts.mcp_indexer_server")


class FakeQdrant:
    def __init__(self):
        self._collections = {"test": {"vectors_count": 123, "points_count": 456}}

    def get_collections(self):
        return types.SimpleNamespace(collections=[types.SimpleNamespace(name="test")])

    def get_collection(self, collection_name):
        return types.SimpleNamespace(points_count=456, vectors_count=123)


@pytest.mark.service
def test_qdrant_status_mocked(monkeypatch):
    monkeypatch.setenv("COLLECTION_NAME", "test")
    monkeypatch.setattr(srv, "QdrantClient", lambda *a, **k: FakeQdrant())

    out = srv.asyncio.get_event_loop().run_until_complete(srv.qdrant_status())
    assert out["ok"] is True
    assert out["collections"][0]["name"] == "test"

