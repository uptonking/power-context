import os
import json
import uuid
import importlib
import pytest
from pathlib import Path

pytestmark = pytest.mark.integration

ing = importlib.import_module("scripts.ingest_code")
srv = importlib.import_module("scripts.mcp_indexer_server")


class FakeEmbedder:
    def __init__(self, model_name: str = "fake"):
        self.model_name = model_name
    def embed(self, texts):
        # Deterministic small vector by hashing
        out = []
        for t in texts:
            h = sum(ord(c) for c in t) % 997
            vec = [(float((h + i) % 13) / 13.0) for i in range(32)]
            out.append(vec)
        return out


@pytest.fixture(scope="module")
def qdrant_container():
    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.waiting_utils import wait_for_logs
    except Exception as e:  # pragma: no cover
        pytest.skip("testcontainers not available")
    container = DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333)
    container.start()
    try:
        # Wait for Qdrant to be ready
        wait_for_logs(container, "Actix runtime found; starting workers")
    except Exception:
        pass
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(6333))
    url = f"http://{host}:{port}"
    yield url
    container.stop()


@pytest.mark.integration
def test_index_and_search_minirepo(tmp_path, monkeypatch, qdrant_container):
    # Env for services
    os.environ["QDRANT_URL"] = qdrant_container
    os.environ["COLLECTION_NAME"] = f"test-{uuid.uuid4().hex[:8]}"
    os.environ["USE_TREE_SITTER"] = "0"

    # Stub embeddings everywhere
    monkeypatch.setattr(ing, "TextEmbedding", lambda *a, **k: FakeEmbedder("fake"))
    monkeypatch.setattr(srv, "_get_embedding_model", lambda *a, **k: FakeEmbedder("fake"))

    # Create tiny repo
    (tmp_path / "pkg").mkdir()
    f1 = tmp_path / "pkg" / "a.py"
    f1.write_text("def f():\n    return 1\n")
    f2 = tmp_path / "pkg" / "b.txt"
    f2.write_text("hello world\nthis is a test\n")

    # Index via function call (no shell)
    ing.index_repo(root=tmp_path, recreate=True)

    # Search directly via async function
    res = srv.asyncio.get_event_loop().run_until_complete(
        srv.repo_search(
            queries=["def f"],
            limit=5,
            language="python",
            under=str(tmp_path),
            include_snippet=True,
            compact=False,
        )
    )

    assert res.get("ok", True)
    assert any(str(f1) in (r.get("path") or "") for r in res.get("results", []))

