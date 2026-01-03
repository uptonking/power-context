#!/usr/bin/env python3
"""
CoIR Runner for Context-Engine.

This module wires the external `coir-eval` package to Context-Engine's retriever
adapter (`scripts.benchmarks.coir.retriever.ContextEngineRetriever`).

## Context-Engine Features Used

**Search Pipeline (via repo_search):**
- [x] Hybrid search (dense + lexical RRF fusion)
- [x] ONNX reranker (BAAI/bge-reranker-base)
- [x] Query expansion (synonyms via HYBRID_EXPAND=1)
- [x] Semantic expansion (SEMANTIC_EXPANSION_ENABLED=1)

**Indexing Pipeline (benchmark corpus):**
- [x] Dense embeddings (bge-base-en)
- [x] Lexical hash vectors (for hybrid search)
- [x] Chunking (semantic or micro, depending on env)
- [x] Optional AST symbol/import/call enrichment (when supported)
- [x] Optional ReFRAG mini vectors / pattern / sparse vectors
- [x] Corpus fingerprinting (smart reuse)

## Environment Variables

Enable additional features:
    HYBRID_EXPAND=1          # Query expansion (default: on)
    SEMANTIC_EXPANSION_ENABLED=1  # Semantic expansion (default: on)
    HYBRID_IN_PROCESS=1      # In-process hybrid search (default: on)

Reranking:
    RERANK_ENABLED=1         # Enable ONNX reranker (default: on)
    RERANK_IN_PROCESS=1      # Run reranker in-process (required for reliability)
    RERANK_TOP_N=50          # Number of candidates to rerank
    RERANK_RETURN_M=20       # Number of results to return after rerank

Notes:
- CoIR datasets may require network access to download on first run.
- We keep this runner dependency-light: if `coir-eval` isn't installed, we print
  a clear installation hint instead of crashing with an obscure ImportError.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from scripts.benchmarks.coir import DEFAULT_TASKS, COIR_TASKS
from scripts.benchmarks.coir.retriever import ContextEngineRetriever
from scripts.benchmarks.common import get_runtime_info


@dataclass
class CoIRReport:
    """Light wrapper around the results returned by coir-eval."""

    tasks: List[str]
    raw: Any
    runtime_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.runtime_info is None:
            self.runtime_info = get_runtime_info()

    def to_dict(self) -> Dict[str, Any]:
        try:
            # coir-eval often returns dict-like results already
            if isinstance(self.raw, dict):
                payload = dict(self.raw)
            else:
                payload = {"raw": self.raw}
        except Exception:
            payload = {"raw": str(self.raw)}
        return {
            "tasks": list(self.tasks),
            "runtime_info": self.runtime_info,
            "results": payload,
        }


def _ensure_env_defaults() -> None:
    # Align with other benchmark scripts: fix docker hostname and ensure non-empty URL.
    if "qdrant:" in (os.environ.get("QDRANT_URL", "") or ""):
        os.environ["QDRANT_URL"] = "http://localhost:6333"
    if not (os.environ.get("QDRANT_URL") or "").strip():
        os.environ["QDRANT_URL"] = "http://localhost:6333"
    # Enable in-process reranker for reliability
    os.environ.setdefault("RERANK_IN_PROCESS", "1")
    # Determinism defaults (best-effort; still recorded in runtime info)
    os.environ.setdefault("EMBEDDING_SEED", "42")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("QDRANT_EF_SEARCH", "128")

    # Set reranker model paths (relative to project root)
    from pathlib import Path
    _project_root = Path(__file__).parent.parent.parent.parent
    os.environ.setdefault("RERANKER_ONNX_PATH", str(_project_root / "models" / "model_qint8_avx512_vnni.onnx"))
    os.environ.setdefault("RERANKER_TOKENIZER_PATH", str(_project_root / "models" / "tokenizer.json"))

    try:
        seed = int(os.environ.get("EMBEDDING_SEED", "42") or 42)
        import random
        random.seed(seed)
        try:
            import numpy as np  # type: ignore
            np.random.seed(seed)
        except Exception:
            pass
    except Exception:
        pass


def _load_coir_tasks(task_names: List[str], limit: Optional[int] = None) -> Any:
    """
    Load tasks using coir-eval.

    We use signature introspection so this code stays compatible across minor
    upstream API changes (e.g., `limit` vs `max_samples`).
    """
    try:
        from coir.data_loader import get_tasks  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: coir-eval. Install with: pip install coir-eval"
        ) from e

    import inspect

    kwargs: Dict[str, Any] = {"tasks": list(task_names)}
    if limit is not None:
        sig = None
        try:
            sig = inspect.signature(get_tasks)
        except Exception:
            sig = None
        if sig is not None:
            if "limit" in sig.parameters:
                kwargs["limit"] = int(limit)
            elif "max_samples" in sig.parameters:
                kwargs["max_samples"] = int(limit)
            elif "n_samples" in sig.parameters:
                kwargs["n_samples"] = int(limit)

    return get_tasks(**kwargs)


def _safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


def _task_iter(tasks_obj: Any) -> List[Any]:
    if tasks_obj is None:
        return []
    if isinstance(tasks_obj, dict):
        return list(tasks_obj.values())
    if isinstance(tasks_obj, (list, tuple)):
        return list(tasks_obj)
    tasks_attr = getattr(tasks_obj, "tasks", None)
    if isinstance(tasks_attr, (list, tuple)):
        return list(tasks_attr)
    return []


def _get_task_name(task: Any, fallback: str) -> str:
    if isinstance(task, dict):
        return str(
            task.get("name")
            or task.get("task_name")
            or task.get("id")
            or fallback
        )
    for attr in ("name", "task_name", "id", "dataset"):
        val = getattr(task, attr, None)
        if val:
            return str(val)
    return fallback


def _get_task_field(task: Any, names: List[str]) -> Any:
    if isinstance(task, dict):
        for name in names:
            if name in task:
                return task.get(name)
        return None
    for name in names:
        if hasattr(task, name):
            return getattr(task, name)
    return None


def _summarize_task_sizes(tasks_obj: Any) -> List[Dict[str, Optional[int]]]:
    stats: List[Dict[str, Optional[int]]] = []
    if isinstance(tasks_obj, dict):
        task_items = list(tasks_obj.items())
    else:
        task_items = [
            (_get_task_name(task, f"task_{idx}"), task)
            for idx, task in enumerate(_task_iter(tasks_obj))
        ]

    for name, task in task_items:
        if isinstance(task, tuple) and len(task) >= 2:
            corpus_obj, query_obj = task[0], task[1]
        else:
            corpus_obj = _get_task_field(task, ["corpus", "documents", "docs", "docstore"])
            query_obj = _get_task_field(task, ["queries", "query", "questions"])
        stats.append(
            {
                "task": str(name),
                "corpus_size": _safe_len(corpus_obj),
                "query_size": _safe_len(query_obj),
            }
        )
    return stats


def _evaluate_with_custom_search(
    tasks_obj: Any,
    model: ContextEngineRetriever,
    output_folder: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Run evaluation using OUR search() method, not coir-eval's DRES wrapper.

    This is critical: coir-eval's default path uses encode_queries/encode_corpus
    and does its own similarity computation, completely bypassing our hybrid
    search, reranking, and other features.

    This function calls model.search() directly and computes metrics ourselves.
    """
    import json as json_mod
    try:
        from coir.beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as e:
        raise RuntimeError("Missing coir-eval. Install with: pip install coir-eval") from e

    results: Dict[str, Any] = {}
    os.makedirs(output_folder, exist_ok=True)

    # Handle both dict-style and list-style task objects
    task_items: List[Tuple[str, Any]] = []
    if isinstance(tasks_obj, dict):
        task_items = list(tasks_obj.items())
    else:
        # List/tuple of task objects
        for idx, task in enumerate(_task_iter(tasks_obj)):
            name = _get_task_name(task, f"task_{idx}")
            task_items.append((name, task))

    for task_name, task_data in task_items:
        print(f"[coir] Evaluating task: {task_name}", flush=True)
        output_file = os.path.join(output_folder, f"{task_name}.json")

        # Skip if already evaluated
        if os.path.exists(output_file):
            print(f"[coir] Results for {task_name} already exist. Skipping.", flush=True)
            with open(output_file) as f:
                results[task_name] = json_mod.load(f).get("metrics", {})
            continue

        # Extract corpus, queries, qrels from task data
        if isinstance(task_data, tuple) and len(task_data) == 3:
            corpus, queries, qrels = task_data
        else:
            # Task object with attributes
            corpus = _get_task_field(task_data, ["corpus", "documents", "docs", "docstore"])
            queries = _get_task_field(task_data, ["queries", "query", "questions"])
            qrels = _get_task_field(task_data, ["qrels", "relevance", "labels"])

        if corpus is None or queries is None or qrels is None:
            print(f"[coir][warn] Could not extract corpus/queries/qrels for {task_name}", flush=True)
            continue

        # CRITICAL: Call OUR search() method directly
        # This uses hybrid search + reranking - the full Context-Engine pipeline
        print(f"[coir] Running Context-Engine search on {len(queries)} queries...", flush=True)
        task_results = model.search(corpus, queries, top_k=top_k)

        # Use coir's evaluation metrics (NDCG, MAP, Recall, Precision)
        # EvaluateRetrieval.evaluate is a static-ish method that just computes metrics
        dummy_retriever = EvaluateRetrieval(None, score_function="cos_sim")
        ndcg, map_score, recall, precision = dummy_retriever.evaluate(
            qrels, task_results, dummy_retriever.k_values
        )

        metrics = {
            "NDCG": ndcg,
            "MAP": map_score,
            "Recall": recall,
            "Precision": precision,
        }

        # Save results
        with open(output_file, "w") as f:
            json_mod.dump({"metrics": metrics, "pipeline": "context-engine-hybrid"}, f, indent=2)

        print(f"[coir] {task_name}: NDCG@10={ndcg.get('NDCG@10', 'N/A')}", flush=True)
        results[task_name] = metrics

    return results


def run_coir_benchmark_sync(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 64,
    rerank_enabled: bool = True,
    output_folder: Optional[str] = None,
    top_k: int = 10,
    **kwargs: Any,
) -> CoIRReport:
    """
    Run CoIR benchmark (sync).

    Args:
      tasks: list of task names (subset of COIR_TASKS)
      limit: optional sample limit passed to coir task loader (best-effort)
      batch_size: batch size for coir-eval (if supported)
      rerank_enabled: whether repo_search should attempt reranking
      output_folder: where to write coir-eval artifacts (if supported)
    """
    _ensure_env_defaults()

    task_names = tasks or list(DEFAULT_TASKS)
    invalid = [t for t in task_names if t not in COIR_TASKS]
    if invalid:
        raise ValueError(f"Unknown CoIR task(s): {invalid}. Known: {COIR_TASKS}")

    # Load tasks (may download data)
    print(f"[coir] Loading tasks: {task_names} (limit={limit})", flush=True)
    tasks_obj = _load_coir_tasks(task_names, limit=limit)
    task_stats = _summarize_task_sizes(tasks_obj)
    if task_stats:
        print(f"[coir] Task sizes: {task_stats}", flush=True)
    else:
        print("[coir] Task size verification unavailable (unknown task shape).", flush=True)

    runtime_info = get_runtime_info()
    runtime_info["coir_tasks"] = list(task_names)
    runtime_info["coir_limit_requested"] = limit
    runtime_info["coir_task_stats"] = task_stats

    if limit is not None and task_stats:
        limit_warnings: List[str] = []
        for stat in task_stats:
            task_name = stat.get("task") or "unknown"
            for key in ("corpus_size", "query_size"):
                size = stat.get(key)
                if size is not None and size > int(limit * 1.1):
                    limit_warnings.append(
                        f"{task_name} {key}={size} exceeds requested limit={limit}"
                    )
        if limit_warnings:
            runtime_info["coir_limit_warnings"] = limit_warnings
            for msg in limit_warnings:
                print(f"[coir][warn] {msg}", flush=True)

    print("[coir] Tasks loaded. Starting evaluation...", flush=True)

    # Build model adapter with hybrid search enabled
    model = ContextEngineRetriever(
        use_hybrid_search=True,
        rerank_enabled=bool(rerank_enabled),
        batch_size=batch_size,
        **kwargs,
    )

    # Output directory
    out_dir = (output_folder or "bench_results/coir").strip()
    os.makedirs(out_dir, exist_ok=True)

    # CRITICAL: Use our custom evaluation that calls model.search() directly.
    # coir-eval's default COIR.run() wraps models in DRES which only uses
    # encode_queries/encode_corpus and does its own similarity computation,
    # completely bypassing our hybrid search, reranking, and other features.
    raw = _evaluate_with_custom_search(tasks_obj, model, out_dir, top_k=top_k)

    print("[coir] Evaluation complete.", flush=True)
    return CoIRReport(tasks=task_names, raw=raw, runtime_info=runtime_info)


async def run_coir_benchmark(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 64,
    rerank_enabled: bool = True,
    output_folder: Optional[str] = None,
    top_k: int = 10,
    **kwargs: Any,
) -> CoIRReport:
    """Async wrapper for environments that already use asyncio."""
    return await asyncio.to_thread(
        run_coir_benchmark_sync,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
        rerank_enabled=rerank_enabled,
        output_folder=output_folder,
        top_k=top_k,
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CoIR Benchmark Runner (Context-Engine)")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Task(s) to run. Default: {DEFAULT_TASKS}. Options: {COIR_TASKS}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (best-effort)")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size (if supported)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve per query")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    parser.add_argument("--output-folder", type=str, default=None, help="Write coir-eval artifacts here")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to this file")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Print JSON")
    args = parser.parse_args()

    # Enable Context-Engine features for accurate benchmarking
    if not args.no_expand:
        os.environ.setdefault("HYBRID_EXPAND", "1")
        os.environ.setdefault("SEMANTIC_EXPANSION_ENABLED", "1")
    os.environ.setdefault("HYBRID_IN_PROCESS", "1")

    report = run_coir_benchmark_sync(
        tasks=args.tasks,
        limit=args.limit,
        batch_size=args.batch_size,
        rerank_enabled=not args.no_rerank,
        output_folder=args.output_folder,
        top_k=args.top_k,
    )

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Saved report to: {args.output}")

    if args.json_out:
        import json

        print(json.dumps(report.to_dict(), indent=2))
    else:
        # Print something human-friendly without assuming a particular coir-eval output schema.
        print("CoIR results:")
        try:
            print(report.to_dict())
        except Exception:
            print(asdict(report))


if __name__ == "__main__":
    main()
