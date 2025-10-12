import importlib
import types
import pytest

srv = importlib.import_module("scripts.mcp_indexer_server")


class FakePoint:
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


class FakeQdrantMem:
    def __init__(self, items):
        self._items = items
    def search(self, **kwargs):
        return self._items


@pytest.mark.service
def test_context_search_blend_compact(monkeypatch):
    # repo_search returns two code hits
    def fake_repo_search(**kwargs):
        return {
            "results": [
                {"score": 0.8, "path": "/x/a.py", "start_line": 1, "end_line": 3},
                {"score": 0.6, "path": "/x/b.py", "start_line": 5, "end_line": 9},
            ]
        }
    monkeypatch.setattr(srv, "repo_search", lambda **kw: fake_repo_search(**kw))

    # Memory fallback via Qdrant: two memory-like points (no path in metadata)
    mem_items = [
        FakePoint(0.9, {"content": "note one", "metadata": {}}),
        FakePoint(0.2, {"content": "note two", "metadata": {}}),
    ]
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: FakeQdrantMem(mem_items))

    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.context_search(
            query="foo bar",
            limit=3,
            per_path=1,
            include_memories=True,
            memory_weight=0.5,
            compact=True,
        )
    )

    assert "results" in res
    # Compact shape: code entries have path+lines; memory entries have content only
    for it in res["results"]:
        if it.get("source") == "code":
            assert "path" in it and "start_line" in it and "end_line" in it
        else:
            assert "content" in it and len(it["content"]) > 0


@pytest.mark.service
def test_context_search_weight_scaling(monkeypatch):
    # repo_search returns one code hit
    def fake_repo_search(**kwargs):
        return {"results": [{"score": 0.5, "path": "/x/a.py", "start_line": 1, "end_line": 3}]}
    monkeypatch.setattr(srv, "repo_search", lambda **kw: fake_repo_search(**kw))

    # One memory hit score=0.9 -> after mw=2.0 becomes 1.8 > code 0.5
    mem_items = [FakePoint(0.9, {"content": "note", "metadata": {}})]
    import qdrant_client
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: FakeQdrantMem(mem_items))

    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.context_search(
            query="foo",
            limit=2,
            per_path=1,
            include_memories=True,
            memory_weight=2.0,
            compact=False,
        )
    )

    assert res["results"][0]["source"] == "memory"
    assert res["results"][0]["score"] > res["results"][1]["score"]

