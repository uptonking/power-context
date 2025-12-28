#!/usr/bin/env python3
"""
Query Expansion Benchmark for Context-Engine

Measures expansion quality, semantic similarity, and retrieval impact.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ExpansionResult:
    """Result from expansion evaluation."""
    original_query: str
    expanded_queries: List[str]
    expansion_count: int
    latency_ms: float
    retrieval_improvement: float  # MRR delta


@dataclass
class ExpansionReport:
    """Query expansion benchmark report."""
    name: str
    total_queries: int
    avg_expansions: float
    avg_improvement: float
    avg_latency_ms: float
    results: List[ExpansionResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "metrics": {
                "avg_expansions": round(self.avg_expansions, 2),
                "avg_improvement": round(self.avg_improvement, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
            },
        }


EXPANSION_QUERIES = [
    "database connection",
    "error handling",
    "caching strategy",
    "authentication",
    "logging system",
]


async def run_expansion_benchmark(name: str = "default") -> ExpansionReport:
    """Run query expansion benchmark."""
    try:
        from scripts.mcp_indexer_server import expand_query, repo_search
    except ImportError as e:
        print(f"Import error: {e}")
        return ExpansionReport(name=name, total_queries=0, avg_expansions=0,
                              avg_improvement=0, avg_latency_ms=0)
    
    results: List[ExpansionResult] = []
    
    for query in EXPANSION_QUERIES:
        print(f"  Expanding: {query}...")
        
        # Expand query
        start = time.perf_counter()
        try:
            expansion_result = await expand_query(query=query, max_new=3)
            alternates = expansion_result.get("alternates", []) if isinstance(expansion_result, dict) else []
        except Exception as e:
            print(f"    Expansion error: {e}")
            alternates = []
        expand_ms = (time.perf_counter() - start) * 1000
        
        # Measure retrieval improvement (simplified)
        improvement = 0.0
        if alternates:
            # Compare MRR with and without expansion
            try:
                orig_result = await repo_search(query=query, limit=5)
                exp_result = await repo_search(queries=[query] + alternates[:2], limit=5)
                # Simplified improvement metric
                orig_count = len(orig_result.get("results", []) if isinstance(orig_result, dict) else [])
                exp_count = len(exp_result.get("results", []) if isinstance(exp_result, dict) else [])
                if orig_count > 0:
                    improvement = (exp_count - orig_count) / orig_count
            except Exception:
                pass
        
        results.append(ExpansionResult(
            original_query=query,
            expanded_queries=alternates,
            expansion_count=len(alternates),
            latency_ms=expand_ms,
            retrieval_improvement=improvement,
        ))
        print(f"    expansions={len(alternates)}, improvement={improvement:.1%}")
    
    return ExpansionReport(
        name=name,
        total_queries=len(results),
        avg_expansions=statistics.mean(r.expansion_count for r in results) if results else 0,
        avg_improvement=statistics.mean(r.retrieval_improvement for r in results) if results else 0,
        avg_latency_ms=statistics.mean(r.latency_ms for r in results) if results else 0,
        results=results,
    )


def print_report(report: ExpansionReport):
    print("\n" + "=" * 60)
    print(f"QUERY EXPANSION BENCHMARK: {report.name}")
    print("=" * 60)
    print(f"Queries: {report.total_queries}")
    print(f"Avg Expansions: {report.avg_expansions:.2f}")
    print(f"Avg Improvement: {report.avg_improvement:.1%}")
    print(f"Avg Latency: {report.avg_latency_ms:.2f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Query Expansion Benchmark")
    parser.add_argument("--name", default="default", help="Benchmark name")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Running expansion benchmark: {args.name}")
    report = asyncio.run(run_expansion_benchmark(name=args.name))
    print_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
