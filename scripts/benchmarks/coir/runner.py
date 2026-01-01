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

**Indexing Pipeline:**
- [x] Dense embeddings (bge-base-en)
- [x] Lexical hash vectors (for hybrid search)
- [x] Corpus fingerprinting (smart reuse)

**Not Applicable (CoIR = document retrieval, not code):**
- N/A AST symbol extraction
- N/A Import/call extraction

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
from typing import Any, Dict, List, Optional

from scripts.benchmarks.coir import DEFAULT_TASKS, COIR_TASKS
from scripts.benchmarks.coir.retriever import ContextEngineRetriever
from scripts.benchmarks.common import get_runtime_info


@dataclass
class CoIRReport:
    """Light wrapper around the results returned by coir-eval."""

    tasks: List[str]
    raw: Any
    runtime_info: Dict[str, Any] = None

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


def run_coir_benchmark_sync(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 64,
    rerank_enabled: bool = True,
    output_folder: Optional[str] = None,
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
    tasks_obj = _load_coir_tasks(task_names, limit=limit)

    # Build model adapter
    model = ContextEngineRetriever(
        use_hybrid_search=True,
        rerank_enabled=bool(rerank_enabled),
        **kwargs,
    )

    # Run evaluation
    try:
        from coir.evaluation import COIR  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: coir-eval. Install with: pip install coir-eval"
        ) from e

    import inspect

    ev_kwargs: Dict[str, Any] = {"tasks": tasks_obj}
    try:
        sig = inspect.signature(COIR)
        if "batch_size" in sig.parameters:
            ev_kwargs["batch_size"] = int(batch_size)
    except Exception:
        # Best-effort; COIR may not accept batch_size in older versions.
        pass

    evaluation = COIR(**ev_kwargs)

    # Some coir-eval versions require output_folder as a positional arg.
    # If caller didn't provide one, default to a stable local folder.
    out_dir = (output_folder or "bench_results/coir").strip()
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        # If we can't create it (permissions), still try letting coir-eval handle it.
        pass

    # Call evaluation.run in a version-tolerant way.
    # Prefer keyword if accepted; otherwise pass as positional.
    raw = None
    try:
        run_sig = inspect.signature(evaluation.run)
    except Exception:
        run_sig = None

    if run_sig is not None and "output_folder" in run_sig.parameters:
        raw = evaluation.run(model, output_folder=out_dir)
    else:
        # Fall back to positional required arg.
        raw = evaluation.run(model, out_dir)

    return CoIRReport(tasks=task_names, raw=raw)


async def run_coir_benchmark(
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 64,
    rerank_enabled: bool = True,
    output_folder: Optional[str] = None,
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
        **kwargs,
    )


def main() -> None:
    import os as _os
    parser = argparse.ArgumentParser(description="CoIR Benchmark Runner (Context-Engine)")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Task(s) to run. Default: {DEFAULT_TASKS}. Options: {COIR_TASKS}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per task (best-effort)")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size (if supported)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    parser.add_argument("--output-folder", type=str, default=None, help="Write coir-eval artifacts here")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Print JSON")
    args = parser.parse_args()

    # Enable Context-Engine features for accurate benchmarking
    if not args.no_expand:
        _os.environ.setdefault("HYBRID_EXPAND", "1")
        _os.environ.setdefault("SEMANTIC_EXPANSION_ENABLED", "1")
    _os.environ.setdefault("HYBRID_IN_PROCESS", "1")

    try:
        report = run_coir_benchmark_sync(
            tasks=args.tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            rerank_enabled=not args.no_rerank,
            output_folder=args.output_folder,
        )
    except Exception as e:
        # Keep this error message helpful for first-time users.
        msg = str(e)
        if "coir-eval" in msg.lower() or "coir." in msg.lower() or "coir " in msg.lower():
            raise
        raise

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


