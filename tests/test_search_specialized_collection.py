import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fn_name", "extra_kwargs"),
    [
        ("_search_tests_for_impl", {"language": "python"}),
        ("_search_config_for_impl", {}),
        ("_search_callers_for_impl", {"language": "python"}),
        ("_search_importers_for_impl", {"language": "python"}),
    ],
)
async def test_specialized_search_forwards_collection(fn_name, extra_kwargs):
    from scripts.mcp_impl import search_specialized as ss

    captured = {}

    async def _fake_repo_search(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "results": []}

    fn = getattr(ss, fn_name)
    await fn(
        query="auth",
        limit=5,
        collection="demo-collection",
        repo_search_fn=_fake_repo_search,
        **extra_kwargs,
    )

    assert captured.get("collection") == "demo-collection"

