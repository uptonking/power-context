#!/usr/bin/env python3
"""
CoSQA Evaluation Runner for Context-Engine Benchmarks.

Runs evaluation against the CoSQA benchmark, computing:
- MRR@k (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)
- Recall@k

## Context-Engine Features Used

**Search Pipeline (via repo_search):**
- [x] Hybrid search (dense + lexical RRF fusion)
- [x] ONNX reranker (BAAI/bge-reranker-base)
- [x] Query expansion (synonyms via HYBRID_EXPAND=1)
- [x] Semantic expansion (SEMANTIC_EXPANSION_ENABLED=1)
- [x] Multi-query fusion

**Indexing Pipeline:**
- [x] AST symbol extraction (_extract_symbols)
- [x] Import/call extraction
- [x] Enriched embeddings (symbols + imports + docstring + code)
- [x] Lexical hash vectors (for hybrid search)
- [ ] ReFRAG/micro-chunks (requires REFRAG_MODE=1, off by default)
- [ ] Pattern vectors (requires indexed pattern vectors)

**Not Applicable (CoSQA limitations):**
- N/A Semantic chunking (snippets are atomic units)
- N/A File-level metadata (no real file paths)
- N/A Git metadata (not a real repo)

## Environment Variables

Enable additional features:
    HYBRID_EXPAND=1          # Query expansion (synonyms)
    SEMANTIC_EXPANSION_ENABLED=1  # Semantic similarity expansion
    LLM_EXPAND_MAX=3         # LLM query expansion
    REFRAG_MODE=1            # ReFRAG micro-chunking (if indexed)
    HYBRID_IN_PROCESS=1      # Run hybrid search in-process

Reranking:
    RERANK_ENABLED=1         # Enable ONNX reranker (default: on)
    RERANK_IN_PROCESS=1      # Run reranker in-process (required for reliability)
    RERANK_TOP_N=50          # Number of candidates to rerank
    RERANK_RETURN_M=20       # Number of results to return after rerank

References:
- CoSQA+ paper: https://arxiv.org/abs/2406.11589
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# NOTE: .env loading moved to _load_benchmark_env() to avoid polluting
# environment when this module is imported (e.g., by tests or __init__.py).
# Call _load_benchmark_env() explicitly before running benchmarks.
_ENV_LOADED = False


def _load_benchmark_env() -> None:
    """Load .env file for benchmark config. Safe to call multiple times."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
    except ImportError:
        pass  # dotenv optional

# Critical reranker settings - ensure these are set for proper benchmark scoring
os.environ.setdefault("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
os.environ.setdefault("RERANK_IN_PROCESS", "1")
os.environ.setdefault("RERANK_LEARNING", "0")  # Default off for benchmarks; CLI can override

# Avoid tokenizers fork warning in benchmark runs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scripts.benchmarks.common import percentile, get_runtime_info

# Collection name for CoSQA corpus
DEFAULT_COLLECTION = "cosqa-corpus"

# Baseline MRR from CoSQA+ paper (Table 5)
# https://arxiv.org/abs/2406.11589
#
# Notes on expected performance:
# - Context-Engine uses bge-base-en (general embedding), not code-specific
# - CodeBERT/UniXcoder are code-specific models trained on code search
# - With our hybrid search + rerank, we expect ~0.24-0.30 MRR
# - This beats BM25 (+45-70%) and general embeddings like CodeT5+
# - To match CodeBERT, we'd need a code-specific embedding model
PAPER_BASELINES = {
    "Lucene (BM25)": 0.167,            # CoSQA+ Table 5
    "BoW": 0.065,                      # CoSQA+ Table 5
    "CodeBERT": 0.392,                 # CoSQA+ Table 5 (code-specific)
    "UniXcoder": 0.319,                # CoSQA+ Table 5 (code-specific)
    "CodeT5+ embedding": 0.266,        # CoSQA+ Table 5
    "text-embedding-3-large": 0.393,   # CoSQA+ Table 5 (best, code-specific)
}


@dataclass
class CoSQAQueryResult:
    """Result for a single CoSQA query evaluation."""
    query_id: str
    query_text: str
    relevant_code_ids: List[str]
    retrieved_code_ids: List[str]
    mrr: float
    ndcg_5: float
    ndcg_10: float
    recall_1: float
    recall_5: float
    recall_10: float
    latency_ms: float
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool


@dataclass
class CoSQAReport:
    """CoSQA benchmark report."""
    name: str
    config: Dict[str, Any]
    total_queries: int
    corpus_size: int
    # Aggregate metrics
    mrr: float
    ndcg_5: float
    ndcg_10: float
    recall_1: float
    recall_5: float
    recall_10: float
    hit_rate_1: float
    hit_rate_5: float
    hit_rate_10: float
    # Latency
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    # Per-query results
    results: List[CoSQAQueryResult] = field(default_factory=list)
    # Comparison to baselines
    baseline_comparison: Dict[str, float] = field(default_factory=dict)
    # Runtime info for reproducibility
    runtime_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "corpus_size": self.corpus_size,
            "total_queries": self.total_queries,
            "runtime_info": self.runtime_info,
            "metrics": {
                "mrr": round(self.mrr, 4),
                "ndcg@5": round(self.ndcg_5, 4),
                "ndcg@10": round(self.ndcg_10, 4),
                "recall@1": round(self.recall_1, 4),
                "recall@5": round(self.recall_5, 4),
                "recall@10": round(self.recall_10, 4),
                "hit_rate@1": round(self.hit_rate_1, 4),
                "hit_rate@5": round(self.hit_rate_5, 4),
                "hit_rate@10": round(self.hit_rate_10, 4),
            },
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 2),
                "p50_ms": round(self.p50_latency_ms, 2),
                "p90_ms": round(self.p90_latency_ms, 2),
                "p99_ms": round(self.p99_latency_ms, 2),
            },
            "baseline_comparison": self.baseline_comparison,
            "per_query": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_mrr(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = 1/rank of first relevant item, or 0 if none found.
    """
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_dcg(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        # DCG = sum(rel_i / log2(i+2))
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Compute DCG: relevance is 1 if doc is relevant, 0 otherwise
    relevances = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = compute_dcg(relevances, k)

    # Compute IDCG: ideal ranking has all relevant docs first
    ideal_relevances = [1.0] * min(len(relevant_ids), k)
    idcg = compute_dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_recall(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Compute Recall at k.

    Recall@k = |relevant ∩ retrieved@k| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_ids)


def has_hit(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> bool:
    """Check if any relevant item is in top-k."""
    relevant_set = set(relevant_ids)
    return any(doc_id in relevant_set for doc_id in retrieved_ids[:k])


# ---------------------------------------------------------------------------
# Search Interface
# ---------------------------------------------------------------------------

async def search_cosqa_corpus(
    query: str,
    collection: str = DEFAULT_COLLECTION,
    limit: int = 10,
    rerank_enabled: bool = True,
    mode: str = "hybrid",
) -> Tuple[List[str], float]:
    """Search CoSQA corpus using Context-Engine's repo_search.

    Uses the same search path as CoIR and production for consistency.

    Args:
        query: Natural language query
        collection: Qdrant collection name
        limit: Maximum results to return
        rerank_enabled: Whether to use reranker
        mode: Search mode (passed to repo_search)

    Returns:
        Tuple of (list of code_ids, latency_ms)
    """
    from scripts.mcp_indexer_server import repo_search

    start = time.perf_counter()

    # Use repo_search for consistency with CoIR and production
    # Retrieve larger candidate pool for reranking (100 candidates -> top limit)
    result = await repo_search(
        query=query,
        limit=limit,
        collection=collection,
        rerank_enabled=rerank_enabled,
        rerank_top_n=100 if rerank_enabled else None,  # Retrieve 100 candidates
        rerank_return_m=limit if rerank_enabled else None,  # Rerank down to limit
        mode=mode,
    )

    # Extract code_ids from results (same pattern as CoIR)
    code_ids = []
    for r in result.get("results", []):
        code_id = r.get("code_id") or r.get("doc_id") or (r.get("payload") or {}).get("code_id")
        if code_id:
            code_ids.append(code_id)

    latency_ms = (time.perf_counter() - start) * 1000
    return code_ids, latency_ms


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

async def run_cosqa_benchmark(
    queries: List[Tuple[str, str, List[str]]],
    collection: str = DEFAULT_COLLECTION,
    corpus_size: int = 0,
    limit: int = 10,
    rerank_enabled: bool = True,
    name: str = "cosqa",
    subset_note: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> CoSQAReport:
    """Run CoSQA evaluation on provided queries.

    Args:
        queries: List of (query_id, query_text, relevant_code_ids)
        collection: Qdrant collection name
        corpus_size: Size of indexed corpus
        limit: Max results per query
        rerank_enabled: Whether to use reranker
        name: Benchmark run name
        progress_callback: Optional callback(completed, total)

    Returns:
        CoSQAReport with all metrics
    """
    # Load .env for benchmark config (only when actually running benchmark)
    _load_benchmark_env()

    results: List[CoSQAQueryResult] = []
    latencies: List[float] = []

    config = {
        "collection": collection,
        "limit": limit,
        "rerank_enabled": rerank_enabled,
        "query_count": len(queries),
        # Capture reranker config for reproducibility
        "env": {
            "RERANKER_MODEL": os.environ.get("RERANKER_MODEL", ""),
            "RERANK_IN_PROCESS": os.environ.get("RERANK_IN_PROCESS", ""),
            "RERANK_LEARNING": os.environ.get("RERANK_LEARNING", ""),
            "HYBRID_IN_PROCESS": os.environ.get("HYBRID_IN_PROCESS", ""),
            "HYBRID_EXPAND": os.environ.get("HYBRID_EXPAND", ""),
            "SEMANTIC_EXPANSION_ENABLED": os.environ.get("SEMANTIC_EXPANSION_ENABLED", ""),
        },
    }
    if subset_note:
        config["subset_note"] = subset_note

    print(f"Running CoSQA benchmark: {len(queries)} queries, limit={limit}, rerank={rerank_enabled}")

    for idx, (query_id, query_text, relevant_ids) in enumerate(queries):
        # Search
        retrieved_ids, latency_ms = await search_cosqa_corpus(
            query=query_text,
            collection=collection,
            limit=limit,
            rerank_enabled=rerank_enabled,
        )
        latencies.append(latency_ms)

        # Compute metrics
        mrr = compute_mrr(relevant_ids, retrieved_ids)
        ndcg_5 = compute_ndcg(relevant_ids, retrieved_ids, 5)
        ndcg_10 = compute_ndcg(relevant_ids, retrieved_ids, 10)
        recall_1 = compute_recall(relevant_ids, retrieved_ids, 1)
        recall_5 = compute_recall(relevant_ids, retrieved_ids, 5)
        recall_10 = compute_recall(relevant_ids, retrieved_ids, 10)
        hit_1 = has_hit(relevant_ids, retrieved_ids, 1)
        hit_5 = has_hit(relevant_ids, retrieved_ids, 5)
        hit_10 = has_hit(relevant_ids, retrieved_ids, 10)

        results.append(CoSQAQueryResult(
            query_id=query_id,
            query_text=query_text,
            relevant_code_ids=relevant_ids,
            retrieved_code_ids=retrieved_ids[:10],
            mrr=mrr,
            ndcg_5=ndcg_5,
            ndcg_10=ndcg_10,
            recall_1=recall_1,
            recall_5=recall_5,
            recall_10=recall_10,
            latency_ms=latency_ms,
            hit_at_1=hit_1,
            hit_at_5=hit_5,
            hit_at_10=hit_10,
        ))

        # Progress
        if progress_callback:
            progress_callback(idx + 1, len(queries))
        elif (idx + 1) % 50 == 0:
            print(f"  Evaluated {idx + 1}/{len(queries)} queries")

    # Aggregate metrics
    n = len(results)
    avg_mrr = sum(r.mrr for r in results) / n if n else 0
    avg_ndcg_5 = sum(r.ndcg_5 for r in results) / n if n else 0
    avg_ndcg_10 = sum(r.ndcg_10 for r in results) / n if n else 0
    avg_recall_1 = sum(r.recall_1 for r in results) / n if n else 0
    avg_recall_5 = sum(r.recall_5 for r in results) / n if n else 0
    avg_recall_10 = sum(r.recall_10 for r in results) / n if n else 0
    hit_rate_1 = sum(1 for r in results if r.hit_at_1) / n if n else 0
    hit_rate_5 = sum(1 for r in results if r.hit_at_5) / n if n else 0
    hit_rate_10 = sum(1 for r in results if r.hit_at_10) / n if n else 0

    # Baseline comparison
    baseline_comparison = {}
    for name_b, mrr_b in PAPER_BASELINES.items():
        delta = avg_mrr - mrr_b
        baseline_comparison[name_b] = {
            "paper_mrr": mrr_b,
            "our_mrr": round(avg_mrr, 4),
            "delta": round(delta, 4),
            "improvement": f"{delta/mrr_b*100:+.1f}%" if mrr_b else "N/A",
        }

    report = CoSQAReport(
        name=name,
        config=config,
        total_queries=n,
        corpus_size=corpus_size,
        mrr=avg_mrr,
        ndcg_5=avg_ndcg_5,
        ndcg_10=avg_ndcg_10,
        recall_1=avg_recall_1,
        recall_5=avg_recall_5,
        recall_10=avg_recall_10,
        hit_rate_1=hit_rate_1,
        hit_rate_5=hit_rate_5,
        hit_rate_10=hit_rate_10,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        p50_latency_ms=percentile(latencies, 0.50),
        p90_latency_ms=percentile(latencies, 0.90),
        p99_latency_ms=percentile(latencies, 0.99),
        results=results,
        baseline_comparison=baseline_comparison,
        runtime_info=get_runtime_info(),
    )

    return report



def print_report(report: CoSQAReport) -> None:
    """Print human-readable benchmark report."""
    print("\n" + "=" * 70)
    print(f"CoSQA BENCHMARK REPORT: {report.name}")
    print("=" * 70)
    print(f"Queries: {report.total_queries} | Corpus: {report.corpus_size}")
    print(f"Config: {report.config}")
    if report.config.get("subset_note"):
        print(f"NOTE: {report.config['subset_note']}")

    print("\n" + "-" * 70)
    print("METRICS:")
    print(f"  MRR:        {report.mrr:.4f}")
    print(f"  NDCG@5:     {report.ndcg_5:.4f}  |  NDCG@10:    {report.ndcg_10:.4f}")
    print(f"  Recall@1:   {report.recall_1:.4f}  |  Recall@5:   {report.recall_5:.4f}  |  Recall@10:  {report.recall_10:.4f}")
    print(f"  Hit@1:      {report.hit_rate_1:.4f}  |  Hit@5:      {report.hit_rate_5:.4f}  |  Hit@10:     {report.hit_rate_10:.4f}")

    print("\n" + "-" * 70)
    print("LATENCY:")
    print(f"  Avg: {report.avg_latency_ms:.1f}ms | P50: {report.p50_latency_ms:.1f}ms | P90: {report.p90_latency_ms:.1f}ms | P99: {report.p99_latency_ms:.1f}ms")

    print("\n" + "-" * 70)
    print("BASELINE COMPARISON (MRR):")
    if report.config.get("subset_note"):
        print("  NOTE: Subset results are not comparable to paper baselines.")
    for name, comp in report.baseline_comparison.items():
        print(f"  vs {name}: {comp['paper_mrr']:.3f} → {comp['our_mrr']:.3f} ({comp['improvement']})")

    print("=" * 70)


async def run_full_benchmark(
    split: str = "test",
    collection: str = DEFAULT_COLLECTION,
    limit: int = 10,
    query_limit: Optional[int] = None,
    corpus_limit: Optional[int] = None,
    rerank_enabled: bool = True,
    recreate_index: bool = False,
) -> CoSQAReport:
    """Run complete CoSQA benchmark: download, index, evaluate.

    Args:
        split: Dataset split to use
        collection: Qdrant collection name
        limit: Max results per query
        query_limit: Limit number of queries (for quick testing)
        corpus_limit: Limit corpus size (for quick testing)
        rerank_enabled: Whether to use reranker
        recreate_index: Whether to recreate the index

    Returns:
        CoSQAReport with all metrics
    """
    # Load .env for benchmark config (only when actually running benchmark)
    _load_benchmark_env()

    from scripts.benchmarks.cosqa.dataset import load_cosqa, get_corpus_for_indexing, get_queries_for_evaluation
    from scripts.benchmarks.cosqa.indexer import index_corpus

    # Step 1: Load dataset
    print("\n▶ Step 1: Loading CoSQA dataset...")
    dataset = load_cosqa(split=split)

    # Prepare queries and corpus (corpus-limit uses query-matched subset)
    all_queries = get_queries_for_evaluation(dataset, limit=None)
    desired_queries = all_queries[:query_limit] if query_limit else all_queries
    corpus_cap = corpus_limit

    if corpus_cap:
        corpus_by_id = {entry.code_id: entry for entry in dataset.iter_corpus()}
        selected_queries = []
        selected_ids: set[str] = set()
        skipped_queries = 0

        for query_id, query_text, relevant_ids in desired_queries:
            rel_ids = [rid for rid in relevant_ids if rid in corpus_by_id]
            if not rel_ids:
                continue
            new_ids = [rid for rid in rel_ids if rid not in selected_ids]
            if selected_ids and len(selected_ids) + len(new_ids) > corpus_cap:
                skipped_queries += 1
                continue
            selected_queries.append((query_id, query_text, rel_ids))
            selected_ids.update(new_ids)

        if not selected_queries:
            for query_id, query_text, relevant_ids in desired_queries:
                rel_ids = [rid for rid in relevant_ids if rid in corpus_by_id]
                if rel_ids:
                    selected_queries = [(query_id, query_text, rel_ids)]
                    selected_ids.update(rel_ids)
                    break

        if not selected_queries:
            print("  [WARN] No queries with relevant docs found; evaluation will be empty")
        else:
            if len(selected_ids) > corpus_cap:
                print(
                    f"  [WARN] corpus-limit too small for selected queries; "
                    f"expanding corpus to {len(selected_ids)}"
                )
                corpus_cap = None
            elif skipped_queries or (query_limit and len(selected_queries) < len(desired_queries)):
                print(
                    f"  [WARN] corpus-limit active; selected {len(selected_queries)}/"
                    f"{len(desired_queries)} queries to fit {len(selected_ids)} docs"
                )
            else:
                print(
                    f"  [WARN] corpus-limit active; selected {len(selected_queries)} queries "
                    f"with {len(selected_ids)} relevant docs"
                )

        subset_entries = []
        if selected_ids:
            for entry in dataset.iter_corpus():
                if entry.code_id in selected_ids:
                    subset_entries.append(entry)
        if corpus_cap:
            for entry in dataset.iter_corpus():
                if entry.code_id not in selected_ids:
                    subset_entries.append(entry)
                    if len(subset_entries) >= corpus_cap:
                        break

        corpus = [entry.to_index_payload() for entry in subset_entries]
        queries = selected_queries
    else:
        corpus = get_corpus_for_indexing(dataset)
        queries = desired_queries

    # Step 2: Index corpus
    print("\n▶ Step 2: Indexing corpus...")
    if corpus_cap:
        print(f"  Limited corpus to {len(corpus)} entries")

    # Check if already indexed (use fingerprint matching, not just points_count)
    # The indexer handles fingerprint checking internally and will recreate if needed
    result = index_corpus(corpus, collection=collection, recreate=recreate_index or bool(corpus_limit))
    if result.get("reused"):
        print(f"  Corpus already indexed (fingerprint match, {result.get('indexed', 0)} entries)")
    else:
        print(f"  Indexed {result.get('indexed', 0)} entries ({result.get('skipped', 0)} skipped, {result.get('errors', 0)} errors)")

    # Step 3: Get queries
    print("\n▶ Step 3: Preparing queries...")
    print(f"  {len(queries)} queries with relevance judgments")

    # Step 4: Run evaluation
    print("\n▶ Step 4: Running evaluation...")
    subset_note = None
    if corpus_limit or query_limit:
        parts = []
        if corpus_limit:
            parts.append(f"corpus_limit={corpus_limit}")
        if query_limit:
            parts.append(f"query_limit={query_limit}")
        subset_note = f"subset evaluation ({', '.join(parts)})"
    report = await run_cosqa_benchmark(
        queries=queries,
        collection=collection,
        corpus_size=len(corpus),
        limit=limit,
        rerank_enabled=rerank_enabled,
        name=f"cosqa-{split}",
        subset_note=subset_note,
    )

    return report


def _spawn_learning_worker(collection: str, project_root: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "learning_reranker_worker.py"),
        "--daemon",
        "--collection",
        collection,
    ]
    return subprocess.Popen(cmd, cwd=project_root, env=os.environ.copy())


def main():
    """CLI entrypoint for CoSQA benchmark."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="CoSQA Benchmark for Context-Engine")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help="Qdrant collection name")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max results per query")
    parser.add_argument("--query-limit", type=int, default=None,
                        help="Limit number of queries (for testing)")
    parser.add_argument("--corpus-limit", type=int, default=None,
                        help="Limit corpus size (for testing)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable reranker")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable query expansion")
    parser.add_argument("--recreate", action="store_true",
                        help="Recreate index from scratch")
    parser.add_argument("--learning-worker", action="store_true",
                        help="Spawn learning reranker worker during the run (enables learning + event logging)")
    parser.add_argument("--output", type=str,
                        help="Output JSON file")
    args = parser.parse_args()

    # Enable Context-Engine features for accurate benchmarking
    # These are the features that differentiate us from basic embedding search
    if not args.no_expand:
        os.environ.setdefault("HYBRID_EXPAND", "1")
        os.environ.setdefault("SEMANTIC_EXPANSION_ENABLED", "1")
    os.environ.setdefault("HYBRID_IN_PROCESS", "1")  # Use in-process hybrid search
    os.environ.setdefault("RERANK_IN_PROCESS", "1")  # Use in-process reranker (required)

    # Set reranker model paths (relative to project root)
    _project_root = Path(__file__).parent.parent.parent.parent
    os.environ.setdefault("RERANKER_ONNX_PATH", str(_project_root / "models" / "model_qint8_avx512_vnni.onnx"))
    os.environ.setdefault("RERANKER_TOKENIZER_PATH", str(_project_root / "models" / "tokenizer.json"))

    learning_proc = None
    if args.learning_worker:
        if args.no_rerank:
            print("  [WARN] --learning-worker ignored because --no-rerank is set")
        else:
            os.environ["RERANK_LEARNING"] = "1"
            os.environ["RERANK_EVENTS_ENABLED"] = "1"
            learning_proc = _spawn_learning_worker(args.collection, _project_root)
            print(f"  [learning-worker] Started (pid {learning_proc.pid}) for {args.collection}")

    try:
        report = asyncio.run(run_full_benchmark(
            split=args.split,
            collection=args.collection,
            limit=args.limit,
            query_limit=args.query_limit,
            corpus_limit=args.corpus_limit,
            rerank_enabled=not args.no_rerank,
            recreate_index=args.recreate,
        ))
    finally:
        if learning_proc and learning_proc.poll() is None:
            learning_proc.terminate()
            try:
                learning_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                learning_proc.kill()

    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()
