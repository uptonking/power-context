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

