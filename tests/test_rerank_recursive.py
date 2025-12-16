"""
Tests for the Tiny Recursive Reranker (TRM-inspired).

Validates:
1. Basic reranking functionality
2. Iterative refinement improves results
3. Early stopping works correctly
4. Latent state carryover functions
5. Integration with existing candidates
"""

import pytest
import numpy as np
from typing import List, Dict, Any


# Import the reranker
from scripts.rerank_recursive import (
    RecursiveReranker,
    RefinementState,
    TinyScorer,
    LatentRefiner,
    ConfidenceEstimator,
    rerank_recursive,
    rerank_recursive_inprocess,
)


class TestTinyScorer:
    """Tests for the tiny scoring network."""
    
    def test_forward_shape(self):
        """Scorer should produce correct output shape."""
        scorer = TinyScorer(dim=64, hidden_dim=128)
        
        query_emb = np.random.randn(64).astype(np.float32)
        doc_embs = np.random.randn(5, 64).astype(np.float32)
        z = np.random.randn(64).astype(np.float32)
        
        scores = scorer.forward(query_emb, doc_embs, z)
        
        assert scores.shape == (5,)
        assert scores.dtype == np.float32
    
    def test_forward_deterministic(self):
        """Same inputs should produce same outputs."""
        scorer = TinyScorer(dim=64)
        
        query_emb = np.random.randn(64).astype(np.float32)
        doc_embs = np.random.randn(3, 64).astype(np.float32)
        z = np.random.randn(64).astype(np.float32)
        
        scores1 = scorer.forward(query_emb, doc_embs, z)
        scores2 = scorer.forward(query_emb, doc_embs, z)
        
        np.testing.assert_array_almost_equal(scores1, scores2)


class TestLatentRefiner:
    """Tests for latent state refinement."""
    
    def test_refine_shape(self):
        """Refiner should produce latent of same dimension."""
        refiner = LatentRefiner(dim=64)
        
        z = np.random.randn(64).astype(np.float32)
        query_emb = np.random.randn(64).astype(np.float32)
        doc_embs = np.random.randn(5, 64).astype(np.float32)
        scores = np.random.randn(5).astype(np.float32)
        
        z_refined = refiner.refine(z, query_emb, doc_embs, scores)
        
        assert z_refined.shape == (64,)
    
    def test_refine_normalized(self):
        """Refined latent should be unit normalized."""
        refiner = LatentRefiner(dim=64)
        
        z = np.random.randn(64).astype(np.float32)
        query_emb = np.random.randn(64).astype(np.float32)
        doc_embs = np.random.randn(5, 64).astype(np.float32)
        scores = np.random.randn(5).astype(np.float32)
        
        z_refined = refiner.refine(z, query_emb, doc_embs, scores)
        
        norm = np.linalg.norm(z_refined)
        assert abs(norm - 1.0) < 1e-5


class TestConfidenceEstimator:
    """Tests for early stopping logic."""
    
    def test_no_stop_on_first_iteration(self):
        """Should not stop on first iteration."""
        estimator = ConfidenceEstimator()
        
        state = RefinementState(
            z=np.zeros(64),
            scores=np.array([0.5, 0.3, 0.1]),
            iteration=0
        )
        state.score_history = [state.scores]
        
        assert not estimator.should_stop(state)
    
    def test_stop_on_convergence(self):
        """Should stop when top-k rankings stabilize."""
        estimator = ConfidenceEstimator()
        
        state = RefinementState(
            z=np.zeros(64),
            scores=np.array([0.5, 0.3, 0.1]),
            iteration=2
        )
        # Same scores twice = converged
        state.score_history = [
            np.array([0.5, 0.3, 0.1]),
            np.array([0.5, 0.3, 0.1])
        ]
        
        assert estimator.should_stop(state)


class TestRecursiveReranker:
    """Tests for the main recursive reranker."""
    
    def test_rerank_returns_same_count(self):
        """Reranker should return same number of candidates."""
        reranker = RecursiveReranker(n_iterations=2, dim=64)
        
        candidates = [
            {"path": "a.py", "symbol": "func_a", "code": "def a(): pass"},
            {"path": "b.py", "symbol": "func_b", "code": "def b(): pass"},
            {"path": "c.py", "symbol": "func_c", "code": "def c(): pass"},
        ]
        
        results = reranker.rerank("search query", candidates)
        
        assert len(results) == 3
    
    def test_rerank_adds_metadata(self):
        """Reranked results should have recursive metadata."""
        reranker = RecursiveReranker(n_iterations=2, dim=64)
        
        candidates = [
            {"path": "a.py", "symbol": "func_a", "code": "def a(): pass"},
        ]
        
        results = reranker.rerank("query", candidates)
        
        assert "recursive_score" in results[0]
        assert "recursive_rank" in results[0]
        assert "recursive_iterations" in results[0]
        assert "score_trajectory" in results[0]
    
    def test_rerank_preserves_original_fields(self):
        """Original candidate fields should be preserved."""
        reranker = RecursiveReranker(n_iterations=2, dim=64)
        
        candidates = [
            {"path": "a.py", "symbol": "func_a", "code": "def a(): pass", "custom": "value"},
        ]
        
        results = reranker.rerank("query", candidates)
        
        assert results[0]["path"] == "a.py"
        assert results[0]["symbol"] == "func_a"
        assert results[0]["custom"] == "value"

