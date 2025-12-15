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
from scripts.rerank_recursive import TinyScorer

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
        self._embedder = None

    def _load_checkpoint(self) -> float:
        """Load last processed timestamp from checkpoint file."""
        checkpoint_path = Path(TinyScorer.WEIGHTS_DIR) / f"checkpoint_{self.collection}.json"
        try:
            if checkpoint_path.exists():
                with open(checkpoint_path) as f:
                    return json.load(f).get("last_ts", 0)
        except Exception:
            pass
        return 0

    def _save_checkpoint(self, ts: float):
        """Save last processed timestamp to checkpoint file."""
        checkpoint_path = Path(TinyScorer.WEIGHTS_DIR) / f"checkpoint_{self.collection}.json"
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({"last_ts": ts, "collection": self.collection}, f)
        except Exception:
            pass

    def _get_embedder(self):
        """Get embedding model (lazy load)."""
        if self._embedder is None:
            try:
                from scripts.embedder import get_model
                self._embedder = get_model()
            except Exception:
                pass
        return self._embedder

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        embedder = self._get_embedder()
        if embedder is None:
            # Fallback: random embeddings (not ideal but allows testing)
            return np.random.randn(len(texts), 768).astype(np.float32)
        return embedder.encode(texts)

    def _project_to_dim(self, embs: np.ndarray, target_dim: int = 256) -> np.ndarray:
        """Project embeddings to target dimension."""
        if embs.shape[-1] == target_dim:
            return embs
        # Simple truncation/padding
        if embs.shape[-1] > target_dim:
            return embs[..., :target_dim]
        # Pad with zeros
        pad_width = [(0, 0)] * (len(embs.shape) - 1) + [(0, target_dim - embs.shape[-1])]
        return np.pad(embs, pad_width)

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
        for event in events:
            try:
                query = event.get("query", "")
                candidates = event.get("candidates", [])
                teacher_scores = event.get("teacher_scores")

                if not query or not candidates or not teacher_scores:
                    continue

                # Build doc texts from candidates
                doc_texts = []
                for c in candidates:
                    parts = []
                    if c.get("symbol"):
                        parts.append(str(c["symbol"]))
                    if c.get("path"):
                        parts.append(str(c["path"]))
                    snippet = c.get("snippet", "")
                    if snippet:
                        parts.append(str(snippet)[:500])
                    doc_texts.append(" ".join(parts) if parts else "empty")

                # Encode query and docs
                query_emb = self._encode([query])[0]
                doc_embs = self._encode(doc_texts)
                query_emb = self._project_to_dim(query_emb.reshape(1, -1), self.scorer.dim)[0]
                doc_embs = self._project_to_dim(doc_embs, self.scorer.dim)

                # Learn from teacher
                teacher_arr = np.array(teacher_scores, dtype=np.float32)
                z = query_emb.copy()
                self.scorer.learn_from_teacher(query_emb, doc_embs, z, teacher_arr)

            except Exception as e:
                logger.debug(f"Error processing event: {e}")
                continue


def discover_collections() -> List[str]:
    """Discover collections with pending events."""
    events_dir = Path(RERANK_EVENTS_DIR)
    if not events_dir.exists():
        return []

    collections = []
    for f in events_dir.glob("events_*.ndjson"):
        # Extract collection name from filename
        name = f.stem.replace("events_", "")
        if name:
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
