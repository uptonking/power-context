import os
import uuid
import importlib
import pytest

pytestmark = pytest.mark.integration

ing = importlib.import_module("scripts.ingest_code")
srv = importlib.import_module("scripts.mcp_indexer_server")


class FakeEmbedder:
    def __init__(self, model_name: str = "fake"):
        self.model_name = model_name

    class _Vec:
        def __init__(self, arr):
            self._arr = arr
        def tolist(self):
            return self._arr
        def __len__(self):
            return len(self._arr)

    def embed(self, texts):
        # Deterministic small vector by hashing; yields objects with .tolist()
        for t in texts:
            h = sum(ord(c) for c in t) % 997
            vec = [(float((h + i) % 13) / 13.0) for i in range(32)]
            yield self._Vec(vec)


@pytest.fixture(scope="module")
def qdrant_container():
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    os.environ.setdefault("TESTCONTAINERS_RYUK_TIMEOUT", "0")
    try:
        from testcontainers.core.container import DockerContainer
    except Exception:  # pragma: no cover
        pytest.skip("testcontainers not available")
    import time, urllib.request

    container = DockerContainer("qdrant/qdrant:latest")
    try:
        container.with_env("TESTCONTAINERS_RYUK_DISABLED", "true")
        container.with_env("TESTCONTAINERS_RYUK_TIMEOUT", "0")
        container.with_exposed_ports(6333)
        container.start()
        host = container.get_container_host_ip()
        deadline = time.time() + 30
        port = None
        last_exc = None
        while time.time() < deadline:
            try:
                port = int(container.get_exposed_port(6333))
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(0.25)
        if port is None:
            raise RuntimeError(f"qdrant port mapping unavailable: {last_exc}")
        url = f"http://{host}:{port}"

        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url + "/readyz", timeout=2) as r:
                    if 200 <= r.status < 300:
                        break
            except Exception:
                pass
            time.sleep(1)

        yield url
    finally:
        try:
            container.stop()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tier2_fallback_unconditional_with_language_filter(tmp_path, monkeypatch, qdrant_container):
    # Env for services
    os.environ["QDRANT_URL"] = qdrant_container
    os.environ["COLLECTION_NAME"] = f"test-{uuid.uuid4().hex[:8]}"
    os.environ["USE_TREE_SITTER"] = "0"
    os.environ["HYBRID_IN_PROCESS"] = "1"
    os.environ["EMBEDDING_MODEL"] = "fake"
    os.environ["REFRAG_GATE_FIRST"] = "1"  # ensure Tier-1 gate-first path is active

    # Stub embeddings everywhere
    monkeypatch.setattr(ing, "TextEmbedding", lambda *a, **k: FakeEmbedder("fake"))
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: FakeEmbedder("fake"))

    # Create tiny repo
    (tmp_path / "pkg").mkdir()
    f1 = tmp_path / "pkg" / "a.py"
    f1.write_text("def f():\n    return 1\n")

    # Index via function call (no shell)
    ing.index_repo(
        root=tmp_path,
        qdrant_url=qdrant_container,
        api_key="",
        collection=os.environ["COLLECTION_NAME"],
        model_name="fake",
        recreate=True,
    )

    # Construct a path_glob that excludes the actual file so Tier-1 yields zero
    nonmatch_glob = str(tmp_path / "does_not_match" / "*.py")

    # Call context_answer with a language filter present; Tier‑2 should trigger and return results
    out = await srv.context_answer(
        query="def f",
        limit=3,
        per_path=1,
        include_snippet=False,
        language="python",
        path_glob=nonmatch_glob,
    )

    assert isinstance(out, dict)
    cits = out.get("citations") or []
    # Despite Tier-1 returning zero due to path_glob, Tier-2 relaxed search should find a.py
    assert any(str(f1) in (c.get("path") or "") for c in cits)
