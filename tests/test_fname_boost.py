import os

from scripts.rerank_recursive import _compute_fname_boost


def test_fname_boost_basic_path_match():
    q = "hybrid search fusion scoring algorithm"
    cand = {"path": "/Users/me/project/scripts/hybrid_search.py"}
    assert _compute_fname_boost(q, cand, 0.12) == 0.24


def test_fname_boost_rel_path_fallback():
    q = "hybrid search fusion scoring algorithm"
    cand = {"rel_path": "scripts/hybrid_search.py"}
    assert _compute_fname_boost(q, cand, 0.12) == 0.24


def test_fname_boost_metadata_path_fallback_and_jsonish_query():
    # Simulate quirky clients that send list-like strings
    q = '["hybrid search fusion scoring algorithm"]'
    cand = {"metadata": {"path": "scripts/hybrid_search.py"}}
    assert _compute_fname_boost(q, cand, 0.12) == 0.24


def test_fname_boost_disabled_when_factor_zero():
    q = "hybrid search fusion scoring algorithm"
    cand = {"path": "scripts/hybrid_search.py"}
    assert _compute_fname_boost(q, cand, 0.0) == 0.0


def test_fname_boost_requires_two_tokens():
    q = "hybrid fusion scoring algorithm"  # only 'hybrid' overlaps
    cand = {"path": "scripts/hybrid_search.py"}
    assert _compute_fname_boost(q, cand, 0.12) == 0.0
