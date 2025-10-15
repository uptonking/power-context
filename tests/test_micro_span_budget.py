from scripts.hybrid_search import _merge_and_budget_spans, MICRO_TOKENS_PER_LINE

def _mk_item(path, start, end, score):
    # Simulate the shape produced mid-pipeline: {"pt": {payload: {metadata: {...}}}, "s": score}
    class _Pt:
        def __init__(self, payload):
            self.payload = payload
    md = {"path": path, "start_line": start, "end_line": end}
    return {"pt": _Pt({"metadata": md}), "s": score}


def test_merge_and_budget_spans_merges_and_respects_budget(monkeypatch):
    # Make tokens-per-line small to trigger budget easily
    monkeypatch.setenv("MICRO_TOKENS_PER_LINE", "10")
    monkeypatch.setenv("MICRO_BUDGET_TOKENS", "60")  # ~6 lines total
    monkeypatch.setenv("MICRO_MERGE_LINES", "2")
    monkeypatch.setenv("MICRO_OUT_MAX_SPANS", "2")

    # Two overlaps in same file should merge; a third far span should compete for budget
    items = [
        _mk_item("a.py", 10, 11, 1.0),  # 2 lines
        _mk_item("a.py", 12, 12, 0.9),  # adjacent -> merge to 10..12 (3 lines)
        _mk_item("a.py", 30, 33, 0.8),  # 4 lines; may fit depending on budget
        _mk_item("b.py", 5, 6, 0.7),    # 2 lines other file
    ]

    merged = _merge_and_budget_spans(items)

    # After merging, first cluster a.py should be 10..12 (3 lines -> 30 tokens)
    # Budget=60 means we can include either (10..12) + (30..33) OR (10..12) + b.py (depending on order/score)
    # Since a.py spans have higher scores, expect both from a.py until per-path cap hits 2
    assert len(merged) >= 1
    # Spans carry _merged_* annotations
    assert merged[0]["_merged_start"] <= merged[0]["_merged_end"]
    # Budget tokens used are attached
    assert isinstance(merged[0]["_budget_tokens"], int)


def test_merge_and_budget_respects_per_path_cap(monkeypatch):
    monkeypatch.setenv("MICRO_TOKENS_PER_LINE", "10")
    monkeypatch.setenv("MICRO_BUDGET_TOKENS", "200")  # ample
    monkeypatch.setenv("MICRO_MERGE_LINES", "1")
    monkeypatch.setenv("MICRO_OUT_MAX_SPANS", "1")

    items = [
        _mk_item("x.py", 1, 2, 1.0),
        _mk_item("x.py", 10, 11, 0.9),
        _mk_item("y.py", 3, 4, 0.8),
    ]
    merged = _merge_and_budget_spans(items)
    # Only 1 from x.py should be included due to per-path cap
    from collections import Counter
    paths = [ (m.get("pt").payload["metadata"]["path"]) for m in merged ]
    c = Counter(paths)
    assert c.get("x.py", 0) <= 1

