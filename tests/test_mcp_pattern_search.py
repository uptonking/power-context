import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_pattern_search_nl_toon_min_score_pass_through(monkeypatch):
    """NL + TOON path should pass min_score through without crashing."""
    from scripts import mcp_impl
    from scripts.mcp_impl import pattern_search as mcp

    # Bypass real imports
    mcp._PATTERN_SEARCH_LOADED = True

    captured = {}

    def fake_search_by_pattern_description(**kwargs):
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
