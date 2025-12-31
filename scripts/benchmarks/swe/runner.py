#!/usr/bin/env python3
"""
SWE-bench Retrieval Evaluation Runner.

Evaluates Context-Engine's file retrieval performance on SWE-bench instances.
For each issue, measures whether we retrieve the files that were actually
modified in the ground-truth patch.

## Metrics

- File Recall@k: Fraction of ground-truth files found in top-k
- File Precision@k: Fraction of top-k results that are ground-truth
- MRR: Mean Reciprocal Rank of first correct file
- Pass@k: Fraction of instances where ALL files found in top-k

## Context-Engine Features Exercised

**Indexing Pipeline (via ingest_code.index_repo):**
- [x] AST extraction (symbols, imports, calls) via tree-sitter
- [x] Semantic chunking (preserve function/class boundaries)
- [x] Dense embeddings (bge-base-en-v1.5)
- [x] Lexical hash vectors (for hybrid search)
- [x] Git metadata (commit, author, date)
- [ ] ReFRAG micro-chunks (enable with REFRAG_MODE=1)
- [ ] Pattern vectors (enable with INDEX_PATTERN_VECTORS=1)

**Search Pipeline (via mcp_indexer_server.repo_search):**
- [x] Hybrid search (dense + lexical RRF fusion)
- [x] ONNX reranker (BAAI/bge-reranker-base)
- [x] Query expansion (synonyms + semantic)
- [x] Symbol boosting (HYBRID_SYMBOL_BOOST)
- [x] Path-based scoring (core file boost, vendor penalty)
- [x] Recency weighting (RECENCY_WEIGHT)
- [ ] Tier-2 fallback (CTX_MULTI_COLLECTION)
- [ ] LLM query expansion (LLM_EXPAND_MAX)

**Unique to SWE-bench (vs CoSQA/CoIR):**
- Real repository structure with cross-file dependencies
- Import graph traversal
- Full AST with call relationships
- Git history context

## Environment Variables (Tuning Knobs)

Hybrid Search Weights:
    HYBRID_RRF_K=30              # RRF constant (higher = more uniform)
    HYBRID_DENSE_WEIGHT=1.5      # Weight for dense vectors
    HYBRID_LEXICAL_WEIGHT=0.20   # Weight for lexical vectors
    HYBRID_SYMBOL_BOOST=0.15     # Boost for symbol matches
    HYBRID_SYMBOL_EQUALITY_BOOST=0.25  # Boost for exact symbol match

Scoring Adjustments:
    RECENCY_WEIGHT=0.1           # Boost recent files
    CORE_FILE_BOOST=0.08         # Boost core paths (src/, lib/)
    VENDOR_PENALTY=0.25          # Penalty for vendor/node_modules
    TEST_FILE_PENALTY=0.15       # Penalty for test files
    IMPLEMENTATION_BOOST=0.05    # Boost non-interface files

Query Expansion:
    HYBRID_EXPAND=1              # Enable query expansion
    SEMANTIC_EXPANSION_ENABLED=1 # Enable semantic expansion
    SEMANTIC_EXPANSION_MAX_TERMS=3  # Max expansion terms
    LLM_EXPAND_MAX=0             # LLM-based expansion (requires API)

Indexing:
    USE_TREE_SITTER=1            # Enable tree-sitter AST
    INDEX_USE_ENHANCED_AST=1     # Advanced AST chunking
    INDEX_SEMANTIC_CHUNKS=1      # Semantic chunking
    INDEX_CHUNK_LINES=120        # Lines per chunk
    INDEX_CHUNK_OVERLAP=20       # Overlap between chunks
    REFRAG_MODE=0                # ReFRAG micro-chunking

Reranking:
    RERANK_ENABLED=1             # Enable reranker
    RERANK_TOP_N=50              # Candidates to rerank

## Usage

    # Quick test (10 instances)
    python -m scripts.benchmarks.swe.runner --subset lite --limit 10

    # Single repo focus
    python -m scripts.benchmarks.swe.runner --repo django/django --limit 20

    # Full evaluation with tuning
    HYBRID_SYMBOL_BOOST=0.25 REFRAG_MODE=1 \\
        python -m scripts.benchmarks.swe.runner --subset lite

    # Compare with/without features
    python -m scripts.benchmarks.swe.runner --limit 50 -o baseline.json
    REFRAG_MODE=1 python -m scripts.benchmarks.swe.runner --limit 50 -o refrag.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ---------------------------------------------------------------------------
# Default environment configuration
# These can be overridden by setting env vars before running
# ---------------------------------------------------------------------------

# Query expansion (on by default for best recall)
os.environ.setdefault("HYBRID_EXPAND", "1")
os.environ.setdefault("SEMANTIC_EXPANSION_ENABLED", "1")
os.environ.setdefault("SEMANTIC_EXPANSION_MAX_TERMS", "3")

# In-process search (faster, no subprocess)
os.environ.setdefault("HYBRID_IN_PROCESS", "1")

# Tree-sitter AST extraction (critical for code understanding)
os.environ.setdefault("USE_TREE_SITTER", "1")
os.environ.setdefault("INDEX_USE_ENHANCED_AST", "1")
os.environ.setdefault("INDEX_SEMANTIC_CHUNKS", "1")

# Reranking (on by default)
os.environ.setdefault("RERANK_ENABLED", "1")


@dataclass
class InstanceResult:
    """Result for a single SWE-bench instance."""
    instance_id: str
    repo: str
    ground_truth_files: list[str]
    retrieved_files: list[str]
    
    # Metrics for this instance
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    all_found: bool = False
    
    # Timing
    index_time_s: float = 0.0
    search_time_s: float = 0.0
    
    error: Optional[str] = None


def get_config_snapshot() -> dict[str, str]:
    """Capture current environment config for reproducibility."""
    keys = [
        # Hybrid search weights
        "HYBRID_RRF_K", "HYBRID_DENSE_WEIGHT", "HYBRID_LEXICAL_WEIGHT",
        "HYBRID_SYMBOL_BOOST", "HYBRID_SYMBOL_EQUALITY_BOOST",
        # Scoring
        "RECENCY_WEIGHT", "CORE_FILE_BOOST", "VENDOR_PENALTY",
        "TEST_FILE_PENALTY", "IMPLEMENTATION_BOOST",
        # Query expansion
        "HYBRID_EXPAND", "SEMANTIC_EXPANSION_ENABLED",
        "SEMANTIC_EXPANSION_MAX_TERMS", "LLM_EXPAND_MAX",
        # Indexing
        "USE_TREE_SITTER", "INDEX_USE_ENHANCED_AST", "INDEX_SEMANTIC_CHUNKS",
        "INDEX_CHUNK_LINES", "INDEX_CHUNK_OVERLAP",
        "REFRAG_MODE", "INDEX_MICRO_CHUNKS",
        # Reranking
        "RERANK_ENABLED", "RERANK_TOP_N",
        # Model
        "EMBEDDING_MODEL",
    ]
    return {k: os.environ.get(k, "") for k in keys if os.environ.get(k)}


@dataclass
class SWEReport:
    """Aggregated SWE-bench retrieval results."""
    subset: str
    total_instances: int
    evaluated_instances: int
    top_k: int

    # Aggregate metrics
    mean_recall: float = 0.0
    mean_precision: float = 0.0
    mrr: float = 0.0
    pass_at_k: float = 0.0

    # Timing
    total_time_s: float = 0.0
    mean_index_time_s: float = 0.0
    mean_search_time_s: float = 0.0

    # Per-instance results
    results: list[InstanceResult] = field(default_factory=list)

    # Errors
    errors: int = 0

    # Config snapshot for reproducibility
    config: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        d = asdict(self)
        # Truncate results for readability
        if len(d["results"]) > 20:
            d["results_sample"] = d["results"][:20]
            d["results_count"] = len(d["results"])
            del d["results"]
        return d


def compute_metrics(
    ground_truth: list[str],
    retrieved: list[str],
) -> tuple[float, float, float, bool]:
    """Compute retrieval metrics for a single instance.
    
    Returns:
        (recall, precision, reciprocal_rank, all_found)
    """
    gt_set = set(ground_truth)
    retrieved_set = set(retrieved)
    
    # How many ground-truth files did we find?
    found = gt_set & retrieved_set
    
    recall = len(found) / len(gt_set) if gt_set else 0.0
    precision = len(found) / len(retrieved) if retrieved else 0.0
    
    # Reciprocal rank: 1/position of first correct file
    rr = 0.0
    for i, f in enumerate(retrieved):
        if f in gt_set:
            rr = 1.0 / (i + 1)
            break
    
    all_found = gt_set <= retrieved_set
    
    return recall, precision, rr, all_found


async def evaluate_instance(
    instance,  # SWEInstance
    repo_manager,  # RepoManager
    top_k: int = 20,
    collection_prefix: str = "swe-bench-",
    rerank_enabled: bool = True,  # noqa: ARG001 - reserved for future use
) -> InstanceResult:
    """Evaluate retrieval for a single SWE-bench instance."""
    result = InstanceResult(
        instance_id=instance.instance_id,
        repo=instance.repo,
        ground_truth_files=instance.ground_truth_files,
        retrieved_files=[],
    )
    
    try:
        # 1. Checkout the correct commit
        repo_path = repo_manager.checkout_commit(
            instance.repo,
            instance.base_commit,
        )
        
        # 2. Index the repository
        t0 = time.perf_counter()
        collection_name = f"{collection_prefix}{instance.repo.replace('/', '-')}"
        
        # Import indexer
        from scripts import ingest_code
        
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        ingest_code.index_repo(
            root=repo_path,
            qdrant_url=qdrant_url,
            api_key=os.environ.get("QDRANT_API_KEY", ""),
            collection=collection_name,
            model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            recreate=False,  # Reuse if same commit
        )
        result.index_time_s = time.perf_counter() - t0

        # 3. Search using the issue text
        t0 = time.perf_counter()

        # Use repo_search from mcp_indexer_server
        from scripts import mcp_indexer_server as srv

        # Set collection for this repo
        old_collection = os.environ.get("COLLECTION_NAME", "")
        os.environ["COLLECTION_NAME"] = collection_name

        try:
            search_result = await srv.repo_search(
                queries=[instance.problem_statement],
                limit=top_k * 2,  # Get more, then dedupe by file
                include_snippet=False,
                compact=True,
            )
        finally:
            if old_collection:
                os.environ["COLLECTION_NAME"] = old_collection

        result.search_time_s = time.perf_counter() - t0

        # 4. Extract unique file paths from results
        seen_files = set()
        for r in search_result.get("results", []):
            path = r.get("path", "")
            if path and path not in seen_files:
                # Normalize path relative to repo root
                if str(repo_path) in path:
                    path = path.replace(str(repo_path) + "/", "")
                seen_files.add(path)
                result.retrieved_files.append(path)
                if len(result.retrieved_files) >= top_k:
                    break

        # 5. Compute metrics
        recall, precision, rr, all_found = compute_metrics(
            instance.ground_truth_files,
            result.retrieved_files,
        )
        result.recall_at_k = recall
        result.precision_at_k = precision
        result.reciprocal_rank = rr
        result.all_found = all_found

    except Exception as e:
        result.error = str(e)

    return result


async def run_full_benchmark(
    subset: str = "lite",
    limit: Optional[int] = None,
    repos: Optional[list[str]] = None,
    top_k: int = 20,
    rerank_enabled: bool = True,
    cache_dir: Optional[str] = None,
) -> SWEReport:
    """Run full SWE-bench retrieval evaluation."""
    from scripts.benchmarks.swe.dataset import load_swe_bench, filter_by_repo
    from scripts.benchmarks.swe.repo_manager import RepoManager

    print("=" * 60)
    print("SWE-bench Retrieval Evaluation")
    print("=" * 60)

    # Load dataset
    instances = load_swe_bench(subset=subset)

    if repos:
        instances = filter_by_repo(instances, repos)
        print(f"Filtered to {len(instances)} instances from {repos}")

    if limit and limit < len(instances):
        instances = instances[:limit]
        print(f"Limited to {limit} instances")

    # Initialize repo manager
    cache_path = Path(cache_dir) if cache_dir else None
    repo_manager = RepoManager(cache_dir=cache_path)

    report = SWEReport(
        subset=subset,
        total_instances=len(instances),
        evaluated_instances=0,
        top_k=top_k,
        config=get_config_snapshot(),
    )

    # Print active config
    print("\nActive configuration:")
    for k, v in report.config.items():
        print(f"  {k}={v}")

    t_start = time.perf_counter()

    # Evaluate each instance
    for i, instance in enumerate(instances):
        print(f"\n[{i+1}/{len(instances)}] {instance.instance_id}")
        print(f"  Files: {instance.ground_truth_files}")

        result = await evaluate_instance(
            instance,
            repo_manager,
            top_k=top_k,
            rerank_enabled=rerank_enabled,
        )

        report.results.append(result)
        report.evaluated_instances += 1

        if result.error:
            report.errors += 1
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Recall: {result.recall_at_k:.2%}, RR: {result.reciprocal_rank:.3f}")

    report.total_time_s = time.perf_counter() - t_start

    # Aggregate metrics
    valid_results = [r for r in report.results if not r.error]
    if valid_results:
        report.mean_recall = sum(r.recall_at_k for r in valid_results) / len(valid_results)
        report.mean_precision = sum(r.precision_at_k for r in valid_results) / len(valid_results)
        report.mrr = sum(r.reciprocal_rank for r in valid_results) / len(valid_results)
        report.pass_at_k = sum(1 for r in valid_results if r.all_found) / len(valid_results)
        report.mean_index_time_s = sum(r.index_time_s for r in valid_results) / len(valid_results)
        report.mean_search_time_s = sum(r.search_time_s for r in valid_results) / len(valid_results)

    return report


def print_report(report: SWEReport):
    """Print formatted report to stdout."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Subset: {report.subset}")
    print(f"Instances: {report.evaluated_instances}/{report.total_instances}")
    print(f"Top-k: {report.top_k}")
    print(f"Errors: {report.errors}")
    print()
    print(f"Mean Recall@{report.top_k}: {report.mean_recall:.2%}")
    print(f"Mean Precision@{report.top_k}: {report.mean_precision:.2%}")
    print(f"MRR: {report.mrr:.4f}")
    print(f"Pass@{report.top_k}: {report.pass_at_k:.2%}")
    print()
    print(f"Total time: {report.total_time_s:.1f}s")
    print(f"Avg index time: {report.mean_index_time_s:.1f}s")
    print(f"Avg search time: {report.mean_search_time_s:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="SWE-bench retrieval evaluation")
    parser.add_argument("--subset", choices=["lite", "full"], default="lite")
    parser.add_argument("--limit", type=int, help="Limit instances")
    parser.add_argument("--repo", action="append", dest="repos", help="Filter to repo(s)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--cache-dir", type=str)
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")

    args = parser.parse_args()

    report = asyncio.run(run_full_benchmark(
        subset=args.subset,
        limit=args.limit,
        repos=args.repos,
        top_k=args.top_k,
        rerank_enabled=not args.no_rerank,
        cache_dir=args.cache_dir,
    ))

    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

