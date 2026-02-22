import pytest


@pytest.mark.asyncio
async def test_symbol_graph_under_uses_path_prefix_matchvalue():
    # Import internal helper to validate filter construction without needing a real Qdrant instance.
    from qdrant_client import models as qmodels
    from scripts.mcp_impl import symbol_graph as sg

    captured = {}

    class FakeClient:
        def scroll(self, *, collection_name, scroll_filter, limit, with_payload, with_vectors):
            captured["collection_name"] = collection_name
            captured["scroll_filter"] = scroll_filter
            return ([], None)

    await sg._query_array_field(  # type: ignore[attr-defined]
        client=FakeClient(),
        collection="codebase",
        field_key="metadata.calls",
        value="foo",
        limit=10,
        language="python",
        under=sg._norm_under("scripts"),  # type: ignore[attr-defined]
    )

    flt = captured.get("scroll_filter")
    assert isinstance(flt, qmodels.Filter)
    must = list(flt.must or [])
    keys = [getattr(c, "key", None) for c in must]
    assert "metadata.path_prefix" in keys

    # Ensure it's an exact match (MatchValue), not substring (MatchText)
    cond = next(c for c in must if getattr(c, "key", None) == "metadata.path_prefix")
    assert isinstance(cond.match, qmodels.MatchValue)
    assert cond.match.value == "/work/scripts"


@pytest.mark.asyncio
async def test_symbol_graph_invalid_query_type_does_not_require_qdrant_import(monkeypatch):
    import builtins
    from scripts.mcp_impl import symbol_graph as sg

    original_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            raise AssertionError("qdrant_client import should not happen for invalid query_type")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    result = await sg._symbol_graph_impl(symbol="foo", query_type="invalid")
    assert "Invalid query_type" in str(result.get("error", ""))
    assert result.get("query_type") == "invalid"


@pytest.mark.asyncio
async def test_symbol_graph_called_by_uses_compute_called_by(monkeypatch):
    import sys
    import types
    from scripts.mcp_impl import symbol_graph as sg

    fake_mod = types.ModuleType("qdrant_client")

    class _FakeQdrantClient:
        def __init__(self, *args, **kwargs):
            pass

    fake_mod.QdrantClient = _FakeQdrantClient
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_mod)

    called = {}

    async def _fake_compute_called_by(symbol, limit=50, language=None, under=None, collection=None):
        called.update(
            {
                "symbol": symbol,
                "limit": limit,
                "language": language,
                "under": under,
                "collection": collection,
            }
        )
        return {
            "symbol": symbol,
            "called_by": [{"path": "a.py", "symbol": "caller"}],
            "count": 1,
            "collection": collection,
        }

    monkeypatch.setattr(sg, "_compute_called_by", _fake_compute_called_by)

    result = await sg._symbol_graph_impl(
        symbol="target_fn",
        query_type="called_by",
        limit=7,
        language="python",
        under="src",
        collection="demo-coll",
    )

    assert called == {
        "symbol": "target_fn",
        "limit": 7,
        "language": "python",
        "under": "src",
        "collection": "demo-coll",
    }
    assert result.get("query_type") == "called_by"
    assert result.get("count") == 1
    assert result.get("results") == [{"path": "a.py", "symbol": "caller"}]

