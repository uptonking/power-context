#!/usr/bin/env python3
"""
Background Learning Reranker Worker.

Consumes training events from the event log and updates weights per-collection.
This keeps the MCP serving path fast and deterministic.

Features:
- Reads events from NDJSON log files (one per collection)
- Batches teacher scoring to amortize ONNX overhead
- Updates weights atomically (write to .tmp, rename)
- Supports multiple collections with isolated weights
- Can run as a daemon or one-shot

Usage:
    # Run continuously (daemon mode)
    python scripts/learning_reranker_worker.py --daemon

    # Process pending events once and exit
    python scripts/learning_reranker_worker.py --once

    # Process specific collection
    python scripts/learning_reranker_worker.py --collection my-repo
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.rerank_events import (
    RERANK_EVENTS_DIR,
    RERANK_EVENTS_RETENTION_DAYS,
    read_events,
    list_event_files,
    cleanup_old_events,
)
from scripts.rerank_recursive import TinyScorer, RecursiveReranker

# Configuration
BATCH_SIZE = int(os.environ.get("RERANK_LEARNING_BATCH_SIZE", "32"))
POLL_INTERVAL = float(os.environ.get("RERANK_LEARNING_POLL_INTERVAL", "30"))
LEARNING_RATE = float(os.environ.get("RERANK_LEARNING_RATE", "0.001"))


def get_logger():
    """Get logger for worker."""
    try:
        from scripts.logger import get_logger as _get_logger
        return _get_logger(__name__)
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)


logger = get_logger()


class CollectionLearner:
    """Handles learning for a single collection."""

    def __init__(self, collection: str):
        self.collection = collection
        self.scorer = TinyScorer(lr=LEARNING_RATE)
        self.scorer.set_collection(collection)
        self._last_processed_ts = self._load_checkpoint()
        # Reuse the serving reranker's embed + project code path 1:1.
        # We only use its private feature helpers, not its scoring loop.
        self._feature_reranker = RecursiveReranker(
            n_iterations=1,
            dim=self.scorer.dim,
            early_stop=False,
            blend_with_initial=0.0,
        )

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        """Sanitize collection name to prevent path traversal."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _load_checkpoint(self) -> float:
        """Load last processed timestamp from checkpoint file."""
        safe_name = self._sanitize_collection(self.collection)
        checkpoint_path = Path(TinyScorer.WEIGHTS_DIR) / f"checkpoint_{safe_name}.json"
        try:
            if checkpoint_path.exists():
                with open(checkpoint_path) as f:
                    return json.load(f).get("last_ts", 0)
        except Exception:
            pass
        return 0

    def _save_checkpoint(self, ts: float):
        """Save last processed timestamp to checkpoint file."""
        safe_name = self._sanitize_collection(self.collection)
        checkpoint_path = Path(TinyScorer.WEIGHTS_DIR) / f"checkpoint_{safe_name}.json"
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({"last_ts": ts, "collection": self.collection}, f)
        except Exception:
            pass

    def _encode_project(self, texts: List[str]) -> np.ndarray:
        """Encode + project via the serving reranker code path."""
        embs = self._feature_reranker._encode(texts)
        return self._feature_reranker._project_to_dim(embs)

    @staticmethod
    def _pack_doc(candidate: Dict[str, Any], max_chars: int = 500) -> str:
        """Pack candidate into doc text (shared by learning and teacher scoring)."""
        parts = []
        if candidate.get("symbol"):
            parts.append(str(candidate["symbol"]))
        if candidate.get("path"):
            parts.append(str(candidate["path"]))
        code = candidate.get("code") or candidate.get("snippet") or candidate.get("text") or ""
        if code:
            parts.append(str(code)[:max_chars])
        return " ".join(parts) if parts else "empty"

    def process_events(self, limit: int = 1000) -> int:
        """Process pending events and return count processed."""
        events = read_events(self.collection, since_ts=self._last_processed_ts, limit=limit)
        if not events:
            return 0

        processed = 0
        batch_events = []

        for event in events:
            batch_events.append(event)
            if len(batch_events) >= BATCH_SIZE:
                self._learn_from_batch(batch_events)
                processed += len(batch_events)
                batch_events = []

        # Process remaining
        if batch_events:
            self._learn_from_batch(batch_events)
            processed += len(batch_events)

        # Update checkpoint
        if events:
            self._last_processed_ts = max(e.get("ts", 0) for e in events)
            self._save_checkpoint(self._last_processed_ts)

        # Save weights after processing batch (with checkpoint every 100 versions)
        if processed > 0:
            self.scorer._save_weights(checkpoint=True)
            metrics = self.scorer.get_metrics()
            logger.info(
                f"[{self.collection}] Processed {processed} events | "
                f"v{metrics['version']} | lr={metrics['learning_rate']:.6f} | "
                f"avg_loss={metrics['avg_loss']:.4f} | converged={metrics['converged']}"
            )

        return processed

    def _learn_from_batch(self, events: List[Dict[str, Any]]):
        """Learn from a batch of events."""
        # Fill missing teacher scores in batch (amortizes ONNX overhead)
        self._maybe_fill_teacher_scores(events)

        for event in events:
            try:
                query = event.get("query", "")
                candidates = event.get("candidates", [])
                teacher_scores = event.get("teacher_scores")

                if not query or not candidates or not teacher_scores:
                    continue

                # Build doc texts from candidates (same packing as teacher scoring)
                doc_texts = [self._pack_doc(c) for c in candidates]

                # Encode query and docs (1:1 with serving embed+project)
                query_emb = self._encode_project([query])[0]
                doc_embs = self._encode_project(doc_texts)

                # Learn from teacher
                teacher_arr = np.array(teacher_scores, dtype=np.float32)
                z = query_emb.copy()
                self.scorer.learn_from_teacher(query_emb, doc_embs, z, teacher_arr)

            except Exception as e:
                logger.debug(f"Error processing event: {e}")
                continue

    def _maybe_fill_teacher_scores(self, events: List[Dict[str, Any]]):
        """Compute teacher scores for events that don't already have them."""
        try:
            from scripts.rerank_local import rerank_local
        except Exception:
            rerank_local = None

        if rerank_local is None:
            return

        # Gather pairs across events so we can call rerank_local once.
        all_pairs: List[tuple] = []
        slices: List[tuple] = []  # (event_index, start, end)

        for event_index, event in enumerate(events):
            if event.get("teacher_scores"):
                continue

            query = event.get("query", "")
            candidates = event.get("candidates", [])
            if not query or not candidates:
                continue

            start = len(all_pairs)
            for c in candidates:
                doc_text = self._pack_doc(c)  # Same packing as learning path
                all_pairs.append((query, doc_text))
            end = len(all_pairs)
            if end > start:
                slices.append((event_index, start, end))

        if not slices:
            return

        try:
            scores = rerank_local(all_pairs)
        except Exception:
            return

        # Map scores back to each event.
        for event_index, start, end in slices:
            try:
                events[event_index]["teacher_scores"] = list(scores[start:end])
            except Exception:
                continue


def discover_collections() -> List[str]:
    """Discover collections with pending events."""
    events_dir = Path(RERANK_EVENTS_DIR)
    if not events_dir.exists():
        return []

    import re

    collections: List[str] = []
    seen = set()
    for f in events_dir.glob("events_*.ndjson"):
        stem = f.stem  # events_<collection>_<YYYYMMDDHH>
        if not stem.startswith("events_"):
            continue

        rest = stem[len("events_") :]
        # Strip the hour suffix if present (10 digits)
        m = re.match(r"^(?P<name>.+)_(?P<hour>\d{10})$", rest)
        name = m.group("name") if m else rest

        if name and name not in seen:
            seen.add(name)
            collections.append(name)

    return collections


def run_once():
    """Process all pending events once and exit."""
    collections = discover_collections()
    if not collections:
        logger.info("No collections with pending events")
        return

    total = 0
    for coll in collections:
        learner = CollectionLearner(coll)
        processed = learner.process_events()
        total += processed

    logger.info(f"Processed {total} events across {len(collections)} collections")


def run_daemon():
    """Run continuously, polling for new events."""
    logger.info(f"Starting learning worker daemon (poll interval: {POLL_INTERVAL}s)")
    learners: Dict[str, CollectionLearner] = {}
    last_cleanup = 0
    cleanup_interval = 3600  # Cleanup old events hourly

    while True:
        try:
            collections = discover_collections()
            for coll in collections:
                if coll not in learners:
                    learners[coll] = CollectionLearner(coll)
                learners[coll].process_events()

            # Periodic cleanup of old event files
            now = time.time()
            if RERANK_EVENTS_RETENTION_DAYS > 0 and now - last_cleanup > cleanup_interval:
                for coll in collections:
                    deleted = cleanup_old_events(coll, RERANK_EVENTS_RETENTION_DAYS)
                    if deleted > 0:
                        logger.info(f"[{coll}] Cleaned up {deleted} old event files")
                last_cleanup = now

        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error(f"Error in daemon loop: {e}")

        time.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Background learning reranker worker")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    parser.add_argument("--collection", type=str, help="Process specific collection only")
    args = parser.parse_args()

    if args.collection:
        learner = CollectionLearner(args.collection)
        processed = learner.process_events()
        logger.info(f"Processed {processed} events for {args.collection}")
    elif args.daemon:
        run_daemon()
    else:
        run_once()


if __name__ == "__main__":
    main()
