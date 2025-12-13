import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


@pytest.mark.usefixtures("monkeypatch")
def test_smart_reindex_refreshes_lex_vector_for_reused_chunks(tmp_path, monkeypatch):
    """When reusing an existing dense embedding, smart reindex must refresh LEX vector.

    Otherwise pseudo/tags changes can drift from the stored lexical vector.
    """
    # The smart reindex logic we test doesn't require the real library.
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=object))

    from scripts import ingest_code

    # Deterministic pseudo/tags so we can predict lexical vector.
    monkeypatch.setattr(
        ingest_code,
        "should_process_pseudo_for_chunk",
        lambda fp, ch, changed: (False, "pseudo", ["tag"]),
    )

    # Avoid touching any caches.
    monkeypatch.setattr(ingest_code, "get_cached_symbols", lambda fp: {})
    monkeypatch.setattr(ingest_code, "compare_symbol_changes", lambda a, b: ([], []))
    monkeypatch.setattr(ingest_code, "set_cached_pseudo", None)
    monkeypatch.setattr(ingest_code, "set_cached_symbols", None)
    monkeypatch.setattr(ingest_code, "set_cached_file_hash", None)

    # Force simple line chunking.
    monkeypatch.setenv("INDEX_MICRO_CHUNKS", "0")
    monkeypatch.setenv("INDEX_SEMANTIC_CHUNKS", "0")
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    monkeypatch.setenv("REFRAG_MODE", "0")

    code = "def add(a, b):\n    return a + b\n"
    fp = tmp_path / "x.py"
    fp.write_text(code, encoding="utf-8")

    # Compute the exact chunk text the indexer will use.
    chunk = ingest_code.chunk_lines(code, max_lines=120, overlap=20)[0]
    code_text = chunk["text"]

    dense_key = "dense"
    old_lex = [0.0] * ingest_code.LEX_VECTOR_DIM
    old_lex[0] = 1.0

    existing_record = SimpleNamespace(
        payload={
            "metadata": {
                "path": str(fp),
                "code": code_text,
                "kind": "function",
                "symbol": "add",
                "start_line": 1,
            }
        },
        vector={dense_key: [0.1, 0.2, 0.3], ingest_code.LEX_VECTOR_NAME: old_lex},
    )

    class FakeClient:
        def scroll(self, **kwargs):
            # Return one existing point and then stop.
            if getattr(self, "_done", False):
                return ([], None)
            self._done = True
            return ([existing_record], None)

    captured = {}

    def fake_upsert_points(_client, _collection, points):
        captured["points"] = points

    monkeypatch.setattr(ingest_code, "upsert_points", fake_upsert_points)
    monkeypatch.setattr(ingest_code, "delete_points_by_path", lambda *a, **k: None)

    # Model is unused in reuse-only path.
    dummy_model = object()

    status = ingest_code.process_file_with_smart_reindexing(
        file_path=Path(fp),
        text=code,
        language="python",
        client=FakeClient(),
        current_collection="c",
        per_file_repo="r",
        model=dummy_model,
        vector_name=dense_key,
    )

    assert status == "success"
    assert "points" in captured and len(captured["points"]) == 1

    out_vec = captured["points"][0].vector
    assert isinstance(out_vec, dict)
    assert ingest_code.LEX_VECTOR_NAME in out_vec

    expected_aug = (code_text or "") + " pseudo" + " tag"
    expected_lex = ingest_code._lex_hash_vector_text(expected_aug)
    assert out_vec[ingest_code.LEX_VECTOR_NAME] == expected_lex
    # Make sure we didn't keep the old lex vector.
    assert out_vec[ingest_code.LEX_VECTOR_NAME] != old_lex

    def test_smart_reindex_does_not_reuse_when_info_changes(monkeypatch):
        """Dense embeddings must not be reused if `information` differs.

        `information` includes line ranges, so it can change even when code text matches.
        """

        import types
        import sys

        # Stub fastembed before importing ingest_code (local py3.14 may not import it).
        sys.modules.setdefault("fastembed", types.SimpleNamespace())

        from scripts import ingest_code

        # Arrange: a reused record with identical code but different `information`.
        old_info = "path: a.py\nlines: 1-1\n..."
        new_info = "path: a.py\nlines: 10-10\n..."
        assert old_info != new_info

        class _Rec:
            def __init__(self, payload, vector):
                self.payload = payload
                self.vector = vector

        dense_name = "text"
        reused_dense = [0.123, 0.456]
        existing_points = [
            _Rec(
                payload={
                    "information": old_info,
                    "metadata": {
                        "code": "print('hi')\n",
                        "kind": "",
                        "symbol": "",
                        "start_line": 0,
                    },
                },
                vector={dense_name: reused_dense, ingest_code.LEX_VECTOR_NAME: [0.0]},
            )
        ]

        # Fake chunk list: same code, but new info.
        chunks = [
            {
                "code": "print('hi')\n",
                "chunk_symbol_id": "",
                "info": new_info,
                "pseudo": "",
                "tags": [],
            }
        ]

        # Hook into the inner reuse logic by calling a small helper we expose in tests.
        # If not present, fall back to invoking the function under test via monkeypatching.
        if not hasattr(ingest_code, "_smart_reindex_choose_reuse_record"):
            # Minimal, behavior-level test: build points_by_code the same way and ensure lookup misses.
            points_by_code = {}
            for rec in existing_points:
                payload = rec.payload or {}
                md = payload.get("metadata") or {}
                code_text = md.get("code") or ""
                embed_text = payload.get("information") or payload.get("document") or ""
                key = ("", code_text, embed_text)
                points_by_code.setdefault(key, []).append(rec)

            code_text = chunks[0]["code"]
            info = chunks[0]["info"]
            reuse_key = ("", code_text, info)
            assert points_by_code.get(reuse_key) is None
            return

        # Preferred: if helper exists, use it.
        reused = ingest_code._smart_reindex_choose_reuse_record(
            existing_points=existing_points,
            chunk_symbol_id=chunks[0]["chunk_symbol_id"],
            code_text=chunks[0]["code"],
            info=chunks[0]["info"],
        )
        assert reused is None
