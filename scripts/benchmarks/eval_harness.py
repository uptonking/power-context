#!/usr/bin/env python3
"""
Comprehensive Evaluation Harness for Context-Engine

Measures MRR, recall@k, precision, latency across query types and datasets.
Supports A/B testing and generates optimization recommendations.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Shared stats helpers
from scripts.benchmarks.common import percentile

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    """Result from evaluating a single query."""
    query: str
    expected_paths: List[str]
    retrieved_paths: List[str]
    latency_ms: float
    mrr: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    name: str
    total_queries: int
    avg_mrr: float
    avg_recall_5: float
    avg_recall_10: float
    avg_precision_5: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    results: List[QueryResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "metrics": {
                "mrr": round(self.avg_mrr, 4),
                "recall@5": round(self.avg_recall_5, 4),
                "recall@10": round(self.avg_recall_10, 4),
                "precision@5": round(self.avg_precision_5, 4),
            },
            "latency": {
                "p50_ms": round(self.p50_latency_ms, 2),
                "p90_ms": round(self.p90_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
        }


def compute_mrr(expected: List[str], retrieved: List[str]) -> float:
    """Mean Reciprocal Rank."""
    for i, path in enumerate(retrieved):
        if any(exp in path for exp in expected):
            return 1.0 / (i + 1)
    return 0.0


def compute_recall(expected: List[str], retrieved: List[str], k: int) -> float:
    """Recall at k."""
    top_k = retrieved[:k]
    hits = sum(1 for exp in expected if any(exp in r for r in top_k))
    return hits / max(1, len(expected))


def compute_precision(expected: List[str], retrieved: List[str], k: int) -> float:
    """Precision at k."""
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if any(exp in r for exp in expected))
    # If fewer than k results returned, precision should be over the retrieved count.
    return hits / max(1, len(top_k))


# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------
EVAL_QUERIES = [
    {"query": "hybrid search RRF ranking", "expected": ["hybrid/ranking.py"]},
    {"query": "memory store implementation", "expected": ["mcp_impl/memory.py"]},
    {"query": "openlit initialization tracing", "expected": ["openlit_init.py"]},
    {"query": "recursive reranker learning", "expected": ["rerank_recursive"]},
    {"query": "embedder model loading", "expected": ["embedder.py"]},
    {"query": "workspace state persistence", "expected": ["workspace_state.py"]},
    {"query": "symbol graph callers", "expected": ["symbol_graph.py", "mcp_impl"]},
    {"query": "context answer grounded citations", "expected": ["context_answer.py"]},
    {"query": "deduplication request fingerprint", "expected": ["deduplication.py"]},
    {"query": "upload service delta bundle", "expected": ["upload_service.py", "upload_delta"]},
]


# ---------------------------------------------------------------------------
# Evaluation Runner  
# ---------------------------------------------------------------------------
async def run_evaluation(
    name: str = "default",
    queries: Optional[List[Dict]] = None,
    config: Optional[Dict] = None,
) -> EvalReport:
    """Run evaluation harness."""
    queries = queries or EVAL_QUERIES
    config = config or {}
    
    # Import search function
    try:
        from scripts.mcp_indexer_server import repo_search
    except ImportError as e:
        print(f"Failed to import: {e}")
        return EvalReport(name=name, total_queries=0, avg_mrr=0, avg_recall_5=0,
                         avg_recall_10=0, avg_precision_5=0, p50_latency_ms=0,
                         p90_latency_ms=0, p99_latency_ms=0)
    
    results: List[QueryResult] = []
    latencies: List[float] = []
    
    for q in queries:
        query_text = q["query"]
        expected = q["expected"]
        
        start = time.perf_counter()
        try:
            result = await repo_search(
                query=query_text,
                limit=config.get("limit", 10),
                per_path=config.get("per_path", 2),
            )
            success = True
        except Exception as e:
            print(f"Query failed: {query_text}: {e}")
            result = {}
            success = False
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        
        # Extract paths
        retrieved = []
        if isinstance(result, dict):
            results_data = result.get("results", [])
            if isinstance(results_data, list):
                for r in results_data:
                    if isinstance(r, dict) and "path" in r:
                        retrieved.append(r["path"])
            elif isinstance(results_data, str):
                for line in results_data.split("\n"):
                    if "/" in line:
                        parts = line.split(",")
                        if parts:
                            retrieved.append(parts[0].strip())
        
        qr = QueryResult(
            query=query_text,
            expected_paths=expected,
            retrieved_paths=retrieved[:10],
            latency_ms=elapsed_ms,
            mrr=compute_mrr(expected, retrieved),
            recall_at_5=compute_recall(expected, retrieved, 5),
            recall_at_10=compute_recall(expected, retrieved, 10),
            precision_at_5=compute_precision(expected, retrieved, 5),
        )
        results.append(qr)
        print(f"  {query_text[:40]:40} MRR={qr.mrr:.3f} R@5={qr.recall_at_5:.2f}")
    
    # Compute aggregates
    return EvalReport(
        name=name,
        total_queries=len(results),
        avg_mrr=statistics.mean(r.mrr for r in results) if results else 0,
        avg_recall_5=statistics.mean(r.recall_at_5 for r in results) if results else 0,
        avg_recall_10=statistics.mean(r.recall_at_10 for r in results) if results else 0,
        avg_precision_5=statistics.mean(r.precision_at_5 for r in results) if results else 0,
        p50_latency_ms=percentile(latencies, 0.50),
        p90_latency_ms=percentile(latencies, 0.90),
        p99_latency_ms=percentile(latencies, 0.99),
        results=results,
    )


# ---------------------------------------------------------------------------
# A/B Comparison
# ---------------------------------------------------------------------------
def compare_reports(baseline: EvalReport, optimized: EvalReport) -> Dict[str, Any]:
    """Compare two evaluation reports."""
    def delta(a: float, b: float) -> str:
        if a == 0:
            return "N/A"
        pct = ((b - a) / a) * 100
        return f"{pct:+.1f}%"
    
    return {
        "baseline": baseline.to_dict(),
        "optimized": optimized.to_dict(),
        "delta": {
            "mrr": delta(baseline.avg_mrr, optimized.avg_mrr),
            "recall@5": delta(baseline.avg_recall_5, optimized.avg_recall_5),
            "p90_latency": delta(baseline.p90_latency_ms, optimized.p90_latency_ms),
        },
        "recommendations": generate_recommendations(baseline, optimized),
    }


def generate_recommendations(baseline: EvalReport, optimized: EvalReport) -> List[Dict]:
    """Generate optimization recommendations."""
    recs = []
    
    # MRR improvement
    mrr_delta = optimized.avg_mrr - baseline.avg_mrr
    if mrr_delta > 0.05:
        recs.append({
            "priority": "high",
            "action": "Deploy optimized configuration",
            "impact": f"+{mrr_delta:.1%} MRR improvement",
        })
    elif mrr_delta < -0.05:
        recs.append({
            "priority": "high", 
            "action": "Revert - quality regression detected",
            "impact": f"{mrr_delta:.1%} MRR decrease",
        })
    
    # Latency improvement
    lat_delta = optimized.p90_latency_ms - baseline.p90_latency_ms
    if lat_delta < -50:
        recs.append({
            "priority": "medium",
            "action": "Latency optimization effective",
            "impact": f"{-lat_delta:.0f}ms P90 reduction",
        })
    
    return recs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_report(report: EvalReport):
    print("\n" + "=" * 60)
    print(f"EVALUATION: {report.name}")
    print("=" * 60)
    print(f"Queries: {report.total_queries}")
    print(f"MRR:         {report.avg_mrr:.4f}")
    print(f"Recall@5:    {report.avg_recall_5:.4f}")
    print(f"Recall@10:   {report.avg_recall_10:.4f}")
    print(f"Precision@5: {report.avg_precision_5:.4f}")
    print("-" * 60)
    print(f"P50 Latency: {report.p50_latency_ms:.2f}ms")
    print(f"P90 Latency: {report.p90_latency_ms:.2f}ms")
    print(f"P99 Latency: {report.p99_latency_ms:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Context-Engine Evaluation Harness")
    parser.add_argument("--name", default="default", help="Evaluation run name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--queries", type=str, help="Custom queries JSON file")
    args = parser.parse_args()
    
    queries = EVAL_QUERIES
    if args.queries and Path(args.queries).exists():
        with open(args.queries) as f:
            queries = json.load(f)
    
    print(f"Running evaluation: {args.name}")
    report = asyncio.run(run_evaluation(name=args.name, queries=queries))
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
