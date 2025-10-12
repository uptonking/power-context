import os
import textwrap
import importlib

ing = importlib.import_module("scripts.ingest_code")


def test_chunk_lines_basic_overlap():
    text = "\n".join(f"L{i}" for i in range(1, 31))
    chunks = ing.chunk_lines(text, max_lines=10, overlap=2)
    assert chunks[0]["start"] == 1 and chunks[0]["end"] == 10
    assert chunks[1]["start"] == 9  # 10 - overlap + 1
    assert chunks[-1]["end"] == 30


def test_chunk_lines_empty():
    chunks = ing.chunk_lines("", max_lines=10, overlap=2)
    assert chunks == []


def test_chunk_semantic_fallback_no_ts(monkeypatch):
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    text = "\n".join(f"L{i}" for i in range(1, 26))
    chunks = ing.chunk_semantic(text, language="python", max_lines=8, overlap=3)
    # Should behave like chunk_lines
    chunks2 = ing.chunk_lines(text, max_lines=8, overlap=3)
    assert chunks == chunks2

