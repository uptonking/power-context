import importlib
import types
import pytest


srv = importlib.import_module("scripts.mcp_indexer_server")


def _fake_items():
    return [
        {
            "score": 1.0,
            "path": "/work/foo.py",
            "symbol": "foo",
            "start_line": 10,
            "end_line": 16,
            "span_budgeted": True,
        },
        {
            "score": 0.9,
            "path": "/work/bar.py",
            "symbol": "bar",
            "start_line": 5,
            "end_line": 8,
            "span_budgeted": True,
        },
    ]


@pytest.mark.service
def test_context_answer_happy_path(monkeypatch):
    # Fake retrieval output (already budgeted)
    import scripts.hybrid_search as hs

    monkeypatch.setattr(hs, "run_hybrid_search", lambda **k: _fake_items())

    # Fake decoder
    import scripts.refrag_llamacpp as ref

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

        def generate_with_soft_embeddings(self, prompt: str, max_tokens: int = 256, **kw):
            assert "Sources:" in prompt and "[1]" in prompt
            return "Answer using [1]"

    monkeypatch.setattr(ref, "LlamaCppRefragClient", FakeLlama)
    monkeypatch.setattr(ref, "is_decoder_enabled", lambda: True)

    out = srv.asyncio.get_event_loop().run_until_complete(
        srv.context_answer(query="how to do x", limit=2, per_path=1)
    )

    assert isinstance(out, dict)
    assert out.get("answer") and "[1]" in out["answer"]
    cits = out.get("citations") or []
    assert len(cits) >= 1
    assert {"path", "start_line", "end_line"}.issubset(set(cits[0].keys()))


def test_context_answer_decoder_disabled(monkeypatch):
    import scripts.hybrid_search as hs

    monkeypatch.setattr(hs, "run_hybrid_search", lambda **k: _fake_items())

    import scripts.refrag_llamacpp as ref

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

        def generate_with_soft_embeddings(self, *a, **k):
            return "SHOULD_NOT_BE_CALLED"

    monkeypatch.setattr(ref, "LlamaCppRefragClient", FakeLlama)
    monkeypatch.setattr(ref, "is_decoder_enabled", lambda: False)

    out = srv.asyncio.get_event_loop().run_until_complete(
        srv.context_answer(query="how to do y", limit=1)
    )

    assert "error" in out
    assert isinstance(out.get("citations"), list)


def test_context_answer_prefers_identifier_spans(monkeypatch):
    import scripts.hybrid_search as hs

    def _items():
        return [
            {
                "score": 1.0,
                "path": "/work/foo.py",
                "symbol": "foo",
                "start_line": 10,
                "end_line": 16,
                "text": "def helper():\n    return 42\n",
            },
            {
                "score": 0.8,
                "path": "/work/bar.py",
                "symbol": "RRF_K",
                "start_line": 5,
                "end_line": 9,
                "text": "RRF_K = 60\n",
            },
        ]

    monkeypatch.setattr(hs, "run_hybrid_search", lambda **k: _items())

    import scripts.refrag_llamacpp as ref

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

        def generate_with_soft_embeddings(self, prompt: str, max_tokens: int = 256, **kw):
            return "Definition: \"RRF_K = 60\" [1]\nUsage: Appears in code. [1]"

    monkeypatch.setattr(ref, "LlamaCppRefragClient", FakeLlama)
    monkeypatch.setattr(ref, "is_decoder_enabled", lambda: True)

    out = srv.asyncio.get_event_loop().run_until_complete(
        srv.context_answer(query="what is RRF_K in hybrid_search.py?", limit=1, per_path=1)
    )

    cits = out.get("citations") or []
    assert len(cits) == 1
    assert cits[0]["path"] == "/work/bar.py"


def test_context_answer_tier2_retry_without_gating(monkeypatch):
    """Tier 2 should retry run_hybrid_search with relaxed filters when Tier 1 yields zero."""

    import scripts.hybrid_search as hs

    calls = []

    def _run_hybrid_search(**kwargs):
        calls.append(kwargs)
        # Only return results once Tier 2 relaxes the filters (path_glob None, symbol None)
        if kwargs.get("path_glob") is None and kwargs.get("symbol") is None and len(calls) >= 1:
            return [
                {
                    "score": 0.42,
                    "path": "/work/hybrid_search.py",
                    "symbol": "RRF_K",
                    "start_line": 100,
                    "end_line": 104,
                    "text": "RRF_K = 60\n",
                }
            ]
        # All other calls (tier1/usage/targeted search) yield no hits
        return []

    monkeypatch.setattr(hs, "run_hybrid_search", _run_hybrid_search)

    import scripts.refrag_llamacpp as ref

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

        def generate_with_soft_embeddings(self, *a, **kw):
            return "Definition: \"RRF_K = 60\" [1]\nUsage: Not found in provided snippets. [1]"

    monkeypatch.setattr(ref, "LlamaCppRefragClient", FakeLlama)
    monkeypatch.setattr(ref, "is_decoder_enabled", lambda: True)

    out = srv.asyncio.get_event_loop().run_until_complete(
        srv.context_answer(query="RRF_K", limit=1, per_path=1)
    )

    # Ensure Tier 2 was invoked (run_hybrid_search called twice)
    assert len(calls) >= 3, "Tier 2 fallback should re-run hybrid search"

    tier2_kwargs = calls[-1]
    # Tier 2 should have relaxed filters
    assert tier2_kwargs.get("path_glob") is None
    assert tier2_kwargs.get("symbol") is None
    assert tier2_kwargs.get("kind") is None

    # The final citations should come from the tier-2 hit
    cits = out.get("citations") or []
    assert len(cits) == 1
    assert cits[0]["path"].endswith("hybrid_search.py")
