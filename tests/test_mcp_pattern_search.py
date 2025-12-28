import pytest
import inspect


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_search_by_pattern_description_accepts_min_score():
    """Verify the real function signature accepts min_score to prevent TypeError."""
    from scripts.pattern_detection.search import search_by_pattern_description

    sig = inspect.signature(search_by_pattern_description)
    params = list(sig.parameters.keys())

    assert "min_score" in params, (
        "search_by_pattern_description must accept min_score kwarg; "
        "MCP wrapper passes it through"
    )


@pytest.mark.anyio
async def test_mcp_pattern_search_nl_toon_min_score_pass_through(monkeypatch):
    """NL + TOON path should pass min_score through without crashing."""
    from scripts.pattern_detection.search import search_by_pattern_description
    from scripts.mcp_impl import pattern_search as mcp

    # Bypass real imports
    mcp._PATTERN_SEARCH_LOADED = True

    captured = {}

    # Use a wrapper that has the REAL signature to catch mismatches
    real_sig = inspect.signature(search_by_pattern_description)
    real_params = set(real_sig.parameters.keys())

    def fake_search_by_pattern_description(**kwargs):
        # Validate only expected kwargs are passed
        unexpected = set(kwargs.keys()) - real_params
        if unexpected:
            raise TypeError(f"Unexpected kwargs: {unexpected}")
        captured.update(kwargs)
        # Simulate TOON output: results is a string, not a list
        return {
            "ok": True,
            "results": "results[0]:",
            "total": 0,
            "search_mode": "natural_language",
        }

    # Avoid code-path; ensure NL branch is chosen
    monkeypatch.setattr(mcp, "_pattern_search_fn", None)
    monkeypatch.setattr(mcp, "_search_by_pattern_description_fn", fake_search_by_pattern_description)

    result = await mcp._pattern_search_impl(
        query="find retry pattern",  # NL triggers description path
        output_format="toon",
        min_score=0.9,
    )

    assert result["ok"] is True
    assert result["query_mode"] == "description"
    # No crash on TOON string results; min_score forwarded to core call
    assert captured["min_score"] == 0.9


@pytest.mark.anyio
async def test_mcp_pattern_search_code_path_aroma_params(monkeypatch):
    """Code path should pass aroma_rerank and aroma_alpha through."""
    from scripts.pattern_detection.search import pattern_search
    from scripts.mcp_impl import pattern_search as mcp

    mcp._PATTERN_SEARCH_LOADED = True

    captured = {}
    real_sig = inspect.signature(pattern_search)
    real_params = set(real_sig.parameters.keys())

    def fake_pattern_search(**kwargs):
        unexpected = set(kwargs.keys()) - real_params
        if unexpected:
            raise TypeError(f"Unexpected kwargs: {unexpected}")
        captured.update(kwargs)
        return {"ok": True, "results": [], "total": 0, "search_mode": "aroma"}

    monkeypatch.setattr(mcp, "_pattern_search_fn", fake_pattern_search)
    monkeypatch.setattr(mcp, "_search_by_pattern_description_fn", None)

    result = await mcp._pattern_search_impl(
        query="for i in range(3): try: pass except: sleep(i)",  # Code path
        aroma_rerank=True,
        aroma_alpha=0.7,
    )

    assert result["ok"] is True
    assert captured["aroma_rerank"] is True
    assert captured["aroma_alpha"] == 0.7
