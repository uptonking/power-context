#!/usr/bin/env python3
"""
Tiny Recursive Reranker - Inspired by TRM (Tiny Recursive Models) paper.

Key innovations from TRM:
1. Iterative refinement: Multiple passes through a tiny network beats one pass through a large network
2. Deep supervision: Train model to *improve* its answer at each step, not predict from scratch
3. Latent state carryover: Maintain a latent vector z that accumulates understanding across iterations
4. Learned early stopping: Stop refining when results are confident enough

Architecture:
- TinyReranker: 2-layer MLP with ~5M params (vs ~100M for typical cross-encoders)
- RecursiveRefinement: Iteratively improve scores by refining latent representation
- ConfidenceGate: Learn when to stop (Q-learning inspired halting)

Usage:
    from scripts.rerank_recursive import RecursiveReranker

    reranker = RecursiveReranker(n_iterations=3)
    reranked = reranker.rerank(query, candidates)
"""

import os
import re
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Very common tokens that appear everywhere - reduce their weight
_COMMON_TOKENS = frozenset({
    "index", "main", "app", "utils", "util", "helper", "helpers", "common",
    "base", "core", "lib", "src", "test", "tests", "spec", "specs",
    "internal", "public", "private", "static", "default", "new", "old",
    "data", "type", "types", "model", "models", "view", "views",
    "the", "and", "for", "with", "from", "that", "this", "have", "are",
})


def _split_identifier(s: str) -> List[str]:
    """Split any identifier into tokens, handling all common conventions.

    Handles: snake_case, kebab-case, camelCase, PascalCase, SCREAMING_CASE,
    numbers, acronyms (XMLParser -> xml, parser), dot.notation, and mixed styles.

    Special handling:
    - Preserves meaningful acronyms (API, HTTP, JSON, XML, URL, etc.)
    - Strips common prefixes (I for interface, _ for private)
    - Handles version suffixes (v2, 2.0)
    """
    if not s:
        return []

    # Strip common prefixes that don't add meaning
    if len(s) > 1:
        # Interface prefix (IUserService -> UserService)
        if s[0] == 'I' and s[1].isupper():
            s = s[1:]
        # Private prefix (_private -> private)
        elif s[0] == '_':
            s = s.lstrip('_')
        # Dollar prefix ($scope -> scope)
        elif s[0] == '$':
            s = s[1:]

    # Insert space before uppercase letters that follow lowercase (camelCase)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # Insert space before uppercase letters followed by lowercase (acronyms: XMLParser -> XML Parser)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Insert space around digit sequences (handler2 -> handler 2, v2 -> v 2)
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)

    # Split on separators: underscore, hyphen, dot, space
    parts = re.split(r"[_\-.\s]+", s)
    tokens = []
    for part in parts:
        part = part.strip().lower()
        # Skip pure numbers and single chars (except meaningful ones)
        if not part:
            continue
        if part.isdigit():
            continue  # Skip version numbers like "2", "18"
        if len(part) < 2:
            continue
        tokens.append(part)

    return tokens


def _normalize_token(tok: str) -> set[str]:
    """Return the token plus simple morphological variants."""
    forms = {tok}
    # Simple plural/singular normalization
    if tok.endswith('s') and len(tok) > 3:
        forms.add(tok[:-1])  # services -> service
    elif tok.endswith('es') and len(tok) > 4:
        forms.add(tok[:-2])  # processes -> process
    elif tok.endswith('ies') and len(tok) > 4:
        forms.add(tok[:-3] + 'y')  # utilities -> utility
    # Add singular -> plural
    if not tok.endswith('s') and len(tok) > 2:
        forms.add(tok + 's')
    return forms


def _tokenize_for_fname_boost(text: Any) -> set[str]:
    """Robust tokenization for filename boosts.

    Some MCP/IDE clients pass query strings that include quotes/brackets
    or list-like wrappers. Regex tokenization is resilient to that.
    """
    if not text:
        return set()
    try:
        s = str(text)
    except Exception:
        return set()

    # Split on any non-alphanumeric
    raw_parts = re.split(r"[^a-zA-Z0-9]+", s)
    tokens = set()
    for part in raw_parts:
        for tok in _split_identifier(part):
            if len(tok) >= 3:  # Query tokens need 3+ chars
                tokens.add(tok)
    return tokens


def _candidate_path_for_fname_boost(candidate: Dict[str, Any]) -> str:
    """Best-effort extraction of a path/filename from candidate objects."""
    for key in ("path", "rel_path", "host_path", "container_path", "client_path"):
        try:
            val = candidate.get(key)
        except Exception:
            val = None
        if isinstance(val, str) and val.strip():
            return val

    try:
        md = candidate.get("metadata") or {}
        if isinstance(md, dict):
            for key in ("path", "rel_path", "host_path", "container_path", "client_path"):
                val = md.get(key)
                if isinstance(val, str) and val.strip():
                    return val
    except Exception:
        pass

    return ""


def _compute_fname_boost(query: Any, candidate: Dict[str, Any], factor: float) -> float:
    """Compute filename/query correlation boost for a candidate.

    Production-grade matching for real-world codebases at scale:

    **Naming convention support:**
    - snake_case, camelCase, PascalCase, kebab-case, SCREAMING_CASE
    - Dot notation (com.company.auth.service)
    - Mixed styles (legacy codebases)

    **Smart tokenization:**
    - Acronyms: XMLParser -> xml, parser; HTTPClient -> http, client
    - Prefixes stripped: IService -> service, _private -> private
    - Numbers separated: handler2 -> handler, React18 -> react

    **Normalization:**
    - Simple plural/singular normalization (services <-> service)

    **Position-aware scoring:**
    - Filename matches weighted higher than directory matches
    - Deeper directories weighted less (noise reduction)

    **Specificity weighting:**
    - Common tokens (index, main, utils) weighted less
    - Rare/specific tokens weighted more

    **Scoring tiers:**
    - Exact match: 1.0 × factor
    - Normalized match (morphology): 0.8 × factor
    - Substring containment: 0.4 × factor
    - Common token penalty: 0.5× multiplier
    - Filename bonus: 1.5× multiplier for filename matches

    Requires 2+ quality matches to trigger (prevents noise).
    """
    if not factor or factor <= 0:
        return 0.0

    query_tokens = _tokenize_for_fname_boost(query)
    if not query_tokens:
        return 0.0

    path = _candidate_path_for_fname_boost(candidate)
    path = str(path or "")
    if not path:
        return 0.0

    # Strip common prefixes that add noise (preserve case for splitting)
    path_clean = path
    path_lower = path.lower()
    for prefix in ("/work/", "/app/", "/src/", "/home/", "/var/", "/opt/", "/usr/"):
        if path_lower.startswith(prefix):
            path_clean = path[len(prefix):]
            break

    # Split path into segments, track position for weighting
    path_segments = re.split(r"[/\\]", path_clean)
    path_segments = [s for s in path_segments if s]  # Remove empty

    if not path_segments:
        return 0.0

    # Tokenize with position info: (token, is_filename, depth)
    # Filename = last segment, depth = 0 for filename, 1 for parent, etc.
    path_token_info: Dict[str, Dict[str, Any]] = {}  # token -> {is_filename, min_depth}

    for i, segment in enumerate(reversed(path_segments)):
        is_filename = (i == 0)
        depth = i

        # Strip extension from filename
        if is_filename and "." in segment:
            segment = segment.rsplit(".", 1)[0]

        for tok in _split_identifier(segment):
            if len(tok) >= 2:
                if tok not in path_token_info:
                    path_token_info[tok] = {"is_filename": is_filename, "depth": depth}
                # Keep the most important occurrence (filename > dir, shallow > deep)
                elif is_filename and not path_token_info[tok]["is_filename"]:
                    path_token_info[tok] = {"is_filename": True, "depth": depth}

    if not path_token_info:
        return 0.0

    path_tokens = set(path_token_info.keys())

    # Build normalized lookup for path tokens
    path_normalized: Dict[str, str] = {}  # normalized_form -> original_token
    for ptok in path_tokens:
        for form in _normalize_token(ptok):
            if form not in path_normalized:
                path_normalized[form] = ptok

    # Score matches with quality tiers
    score = 0.0
    matched_query_tokens = set()

    for qtok in query_tokens:
        qtok_forms = _normalize_token(qtok)
        match_score = 0.0
        matched_ptok = None

        # Tier 1: Exact match
        if qtok in path_tokens:
            match_score = 1.0
            matched_ptok = qtok
        else:
            # Tier 2: Normalized match (plural/singular)
            for qform in qtok_forms:
                if qform in path_normalized:
                    match_score = 0.8
                    matched_ptok = path_normalized[qform]
                    break

            # Tier 3: Substring containment (if no normalized match)
            if match_score == 0.0:
                for ptok in path_tokens:
                    if len(qtok) >= 4 and len(ptok) >= 4:
                        if qtok in ptok or ptok in qtok:
                            overlap = min(len(qtok), len(ptok))
                            if overlap >= 4:
                                match_score = 0.4
                                matched_ptok = ptok
                                break

        if match_score > 0 and matched_ptok:
            matched_query_tokens.add(qtok)

            # Apply position bonus (filename matches worth more)
            info = path_token_info.get(matched_ptok, {})
            if info.get("is_filename"):
                match_score *= 1.5  # 50% bonus for filename match
            else:
                # Depth penalty for deep directories
                depth = info.get("depth", 0)
                if depth > 2:
                    match_score *= 0.8  # Slight penalty for deep paths

            # Common token penalty
            if qtok in _COMMON_TOKENS or matched_ptok in _COMMON_TOKENS:
                match_score *= 0.5

            score += match_score

    # Require 2+ quality matches to trigger (prevents noise from single common word)
    if len(matched_query_tokens) < 2:
        return 0.0

    return float(score * factor)

# Safe imports
try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    HAS_ONNX = True
except ImportError:
    ort = None
    Tokenizer = None
    HAS_ONNX = False


@dataclass
class RefinementState:
    """Carries latent state between refinement iterations."""
    z: np.ndarray  # Latent representation (query understanding)
    scores: np.ndarray  # Current score estimates
    iteration: int = 0
    confidence: float = 0.0  # For early stopping

    # Track per-iteration improvements for analysis
    score_history: List[np.ndarray] = field(default_factory=list)


# Global embedding cache for efficiency
# Key is sha256 hex digest (deterministic, collision-resistant)
_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}
_EMBEDDING_CACHE_MAX_SIZE = 10000
_EMBEDDING_CACHE_LOCK = threading.Lock()


def _cache_key(text: str) -> str:
    """Generate deterministic cache key from text (process-stable, collision-resistant)."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _get_cached_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding from cache if exists."""
    key = _cache_key(text)
    with _EMBEDDING_CACHE_LOCK:
        return _EMBEDDING_CACHE.get(key)


def _cache_embedding(text: str, embedding: np.ndarray):
    """Cache embedding for text."""
    key = _cache_key(text)
    with _EMBEDDING_CACHE_LOCK:
        if len(_EMBEDDING_CACHE) >= _EMBEDDING_CACHE_MAX_SIZE:
            # Evict oldest 10%
            keys_to_remove = list(_EMBEDDING_CACHE.keys())[:_EMBEDDING_CACHE_MAX_SIZE // 10]
            for k in keys_to_remove:
                del _EMBEDDING_CACHE[k]
        _EMBEDDING_CACHE[key] = embedding


class TinyScorer:
    """
    Tiny 2-layer MLP for scoring query-document pairs.

    Inspired by TRM: minimal parameters, maximum iterations.
    Production-ready with:
    - Collection-aware weights with atomic loading
    - Checkpoint versioning (keep last N versions)
    - Training metrics (loss, sample count, convergence)
    - Learning rate decay
    - Hot reload from background worker updates
    """

    # Class-level configuration
    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    WEIGHTS_RELOAD_INTERVAL = float(os.environ.get("RERANKER_WEIGHTS_RELOAD_INTERVAL", "60"))
    MAX_CHECKPOINTS = int(os.environ.get("RERANKER_MAX_CHECKPOINTS", "5"))
    LR_DECAY_STEPS = int(os.environ.get("RERANKER_LR_DECAY_STEPS", "1000"))
    LR_DECAY_RATE = float(os.environ.get("RERANKER_LR_DECAY_RATE", "0.95"))
    MIN_LR = float(os.environ.get("RERANKER_MIN_LR", "0.0001"))

    def __init__(self, dim: int = 256, hidden_dim: int = 512, lr: float = 0.001):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.base_lr = lr
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self._weights_mtime = 0.0
        self._last_reload_check = 0.0

        # Training metrics
        self._update_count = 0
        self._total_samples = 0
        self._cumulative_loss = 0.0
        self._recent_losses: List[float] = []  # Rolling window for convergence detection
        self._version = 0

        # Try to load saved weights, otherwise init random
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
                return
            except Exception as e:
                from scripts.logger import get_logger
                get_logger(__name__).warning(f"TinyScorer: failed to load {self._weights_path}: {e}, using random init")

        self._init_random_weights()

    def _init_random_weights(self):
        """Initialize weights randomly using He initialization (local RNG, deterministic)."""
        # Use local RandomState to avoid polluting global RNG
        rng = np.random.RandomState(42)
        scale = np.float32(np.sqrt(2.0 / (self.dim * 3)))
        self.W1 = rng.randn(self.dim * 3, self.hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        w2_scale = np.float32(np.sqrt(2.0 / self.hidden_dim))
        self.W2 = rng.randn(self.hidden_dim, 1).astype(np.float32) * w2_scale
        self.b2 = np.zeros(1, dtype=np.float32)

        # Momentum for SGD
        self._momentum_W1 = np.zeros_like(self.W1)
        self._momentum_b1 = np.zeros_like(self.b1)
        self._momentum_W2 = np.zeros_like(self.W2)
        self._momentum_b2 = np.zeros_like(self.b2)

    def _update_learning_rate(self):
        """Decay learning rate based on update count."""
        if self._update_count > 0 and self._update_count % self.LR_DECAY_STEPS == 0:
            self.lr = max(self.MIN_LR, self.lr * self.LR_DECAY_RATE)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        avg_loss = self._cumulative_loss / max(1, self._update_count)
        recent_avg = np.mean(self._recent_losses) if self._recent_losses else 0.0
        return {
            "collection": self._collection,
            "version": self._version,
            "update_count": self._update_count,
            "total_samples": self._total_samples,
            "cumulative_loss": self._cumulative_loss,
            "avg_loss": avg_loss,
            "recent_avg_loss": float(recent_avg),
            "learning_rate": self.lr,
            "converged": self._is_converged(),
        }

    def _is_converged(self, window: int = 100, threshold: float = 0.01) -> bool:
        """Check if training has converged (loss not improving)."""
        if len(self._recent_losses) < window:
            return False
        recent = self._recent_losses[-window:]
        first_half = np.mean(recent[:window // 2])
        second_half = np.mean(recent[window // 2:])
        # Converged if improvement is less than threshold
        return abs(first_half - second_half) < threshold * first_half

    def _get_weights_path(self, collection: str) -> str:
        """Get weights file path for a collection."""
        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)
        return os.path.join(self.WEIGHTS_DIR, f"weights_{safe_name}.npz")

    def set_collection(self, collection: str):
        """Set collection and load corresponding weights."""
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def maybe_reload_weights(self):
        """Check if weights file changed and reload if needed (hot reload)."""
        now = time.time()
        if now - self._last_reload_check < self.WEIGHTS_RELOAD_INTERVAL:
            return
        self._last_reload_check = now

        try:
            if os.path.exists(self._weights_path):
                mtime = os.path.getmtime(self._weights_path)
                if mtime > self._weights_mtime:
                    self._load_weights_safe()
        except Exception:
            pass

    def _load_weights_safe(self):
        """Load weights with advisory file locking (prevents partial reads during writes)."""
        import fcntl
        lock_path = self._weights_path + ".lock"
        try:
            # Use same lock file as writer for coordination
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                # Shared lock for reading (blocks if exclusive lock held by writer)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                try:
                    self._load_weights()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            # Fallback to direct load if locking fails
            self._load_weights()

    def forward(self, query_emb: np.ndarray, doc_emb: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Score documents given query and latent state.

        Input shapes:
            query_emb: (dim,) - query embedding
            doc_emb: (n_docs, dim) - document embeddings
            z: (dim,) - latent state from previous iteration

        Returns:
            scores: (n_docs,) - relevance scores
        """
        # Check for hot-reloaded weights
        self.maybe_reload_weights()

        n_docs = doc_emb.shape[0]
        # Broadcast query and z across docs
        q_broadcast = np.tile(query_emb, (n_docs, 1))  # (n_docs, dim)
        z_broadcast = np.tile(z, (n_docs, 1))  # (n_docs, dim)

        # Concatenate [query, doc, latent] for each document
        x = np.concatenate([q_broadcast, doc_emb, z_broadcast], axis=1)  # (n_docs, dim*3)

        # 2-layer MLP with ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)  # (n_docs, hidden_dim)
        scores = (h @ self.W2 + self.b2).squeeze(-1)  # (n_docs,)

        return scores

    def forward_with_cache(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass with cached activations for backprop."""
        z1 = x @ self.W1 + self.b1  # (batch, hidden)
        h1 = np.maximum(0, z1)  # ReLU
        z2 = h1 @ self.W2 + self.b2  # (batch, 1)
        scores = z2.squeeze(-1)  # (batch,)
        cache = {"x": x, "z1": z1, "h1": h1}
        return scores, cache

    def backward(self, dscores: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients."""
        batch_size = dscores.shape[0]
        dz2 = dscores.reshape(-1, 1)  # (batch, 1)

        # Layer 2 gradients
        dW2 = cache["h1"].T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T  # (batch, hidden)

        # ReLU backward
        dz1 = dh1 * (cache["z1"] > 0).astype(np.float32)

        # Layer 1 gradients
        dW1 = cache["x"].T @ dz1
        db1 = dz1.sum(axis=0)

        return {
            "W1": dW1 / batch_size,
            "b1": db1 / batch_size,
            "W2": dW2 / batch_size,
            "b2": db2 / batch_size,
        }

    def learn_from_teacher(
        self,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        z: np.ndarray,
        teacher_scores: np.ndarray,
        margin: float = 0.5,
    ) -> float:
        """
        Online learning: update weights to match ONNX teacher ranking.

        Uses pairwise margin ranking loss:
        L = max(0, margin - (s_pos - s_neg))
        where s_pos > s_neg in teacher ranking.

        Returns:
            Loss value (0.0 if no update needed)
        """
        n_docs = doc_embs.shape[0]
        if n_docs < 2:
            return 0.0

        # Build input matrix
        q_broadcast = np.tile(query_emb, (n_docs, 1))
        z_broadcast = np.tile(z, (n_docs, 1))
        x = np.concatenate([q_broadcast, doc_embs, z_broadcast], axis=1)

        # Forward pass with cache
        our_scores, cache = self.forward_with_cache(x)

        # Get teacher ranking
        teacher_order = np.argsort(-teacher_scores)

        # Sample pairs from teacher ranking (top vs bottom)
        n_pairs = min(5, n_docs // 2)
        total_loss = 0.0
        dscores = np.zeros(n_docs, dtype=np.float32)

        for i in range(n_pairs):
            pos_idx = teacher_order[i]  # Should rank high
            neg_idx = teacher_order[-(i + 1)]  # Should rank low

            # Margin loss
            diff = our_scores[pos_idx] - our_scores[neg_idx]
            if diff < margin:
                # Violation - need to update
                loss = margin - diff
                total_loss += loss

                # Gradient: increase pos score, decrease neg score
                dscores[pos_idx] -= 1.0
                dscores[neg_idx] += 1.0

        # Track metrics
        self._total_samples += n_docs
        self._cumulative_loss += total_loss
        self._recent_losses.append(total_loss)
        if len(self._recent_losses) > 200:  # Keep last 200 for convergence check
            self._recent_losses = self._recent_losses[-200:]

        if total_loss > 0:
            # Backward pass
            grads = self.backward(dscores, cache)

            # SGD with momentum update
            momentum = 0.9
            self._momentum_W1 = momentum * self._momentum_W1 - self.lr * grads["W1"]
            self._momentum_b1 = momentum * self._momentum_b1 - self.lr * grads["b1"]
            self._momentum_W2 = momentum * self._momentum_W2 - self.lr * grads["W2"]
            self._momentum_b2 = momentum * self._momentum_b2 - self.lr * grads["b2"]

            self.W1 += self._momentum_W1
            self.b1 += self._momentum_b1
            self.W2 += self._momentum_W2
            self.b2 += self._momentum_b2

            self._update_count += 1
            self._update_learning_rate()

        return total_loss

    def _save_weights(self, checkpoint: bool = False):
        """
        Save weights to disk atomically (write to .tmp, then rename).

        Uses advisory file locking to coordinate with readers during hot reload.

        Args:
            checkpoint: If True, also save a versioned checkpoint
        """
        import fcntl
        try:
            self._version += 1
            # np.savez automatically adds .npz extension, so use a base path
            # that when .npz is added becomes our tmp file
            tmp_base = self._weights_path.replace(".npz", ".tmp")
            np.savez(
                tmp_base,
                W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                momentum_W1=self._momentum_W1, momentum_b1=self._momentum_b1,
                momentum_W2=self._momentum_W2, momentum_b2=self._momentum_b2,
                update_count=self._update_count,
                total_samples=self._total_samples,
                cumulative_loss=self._cumulative_loss,
                learning_rate=self.lr,
                version=self._version,
                collection=self._collection,
            )
            # np.savez writes to tmp_base + ".npz"
            tmp_path = tmp_base + ".npz"

            # Acquire exclusive lock on target before atomic rename
            # This blocks any readers currently holding shared locks
            lock_path = self._weights_path + ".lock"
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    # Atomic rename to final path
                    os.replace(tmp_path, self._weights_path)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

            # Save versioned checkpoint
            if checkpoint or self._version % 100 == 0:
                self._save_checkpoint()

        except Exception:
            pass

    def _save_checkpoint(self):
        """Save a versioned checkpoint and prune old ones."""
        try:
            checkpoint_path = self._weights_path.replace(".npz", f"_v{self._version}.npz")
            # Copy current weights to checkpoint
            import shutil
            shutil.copy2(self._weights_path, checkpoint_path)

            # Prune old checkpoints (keep last MAX_CHECKPOINTS)
            self._prune_old_checkpoints()
        except Exception:
            pass

    def _prune_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent MAX_CHECKPOINTS."""
        try:
            import glob
            pattern = self._weights_path.replace(".npz", "_v*.npz")
            checkpoints = sorted(glob.glob(pattern))
            if len(checkpoints) > self.MAX_CHECKPOINTS:
                for old_cp in checkpoints[:-self.MAX_CHECKPOINTS]:
                    try:
                        os.remove(old_cp)
                    except Exception:
                        pass
        except Exception:
            pass

    def _load_weights(self):
        """Load weights from disk with dimension validation."""
        from scripts.logger import get_logger
        logger = get_logger(__name__)

        data = np.load(self._weights_path, allow_pickle=True)

        # Helper to safely get from NpzFile (doesn't have .get())
        def _get(key: str, default):
            return data[key] if key in data.files else default

        # Validate all dimensions before loading to prevent shape mismatch crashes
        w1_loaded = data["W1"]
        w2_loaded = data["W2"]
        b1_loaded = data["b1"]
        b2_loaded = data["b2"]

        expected_w1 = (self.dim * 3, self.hidden_dim)
        expected_w2 = (self.hidden_dim, 1)
        expected_b1 = (self.hidden_dim,)
        expected_b2 = (1,)

        shape_ok = (
            w1_loaded.shape == expected_w1 and
            w2_loaded.shape == expected_w2 and
            b1_loaded.shape == expected_b1 and
            b2_loaded.shape == expected_b2
        )

        if not shape_ok:
            logger.warning(
                f"TinyScorer: shape mismatch in {self._weights_path}, "
                f"W1={w1_loaded.shape} (expected {expected_w1}), "
                f"W2={w2_loaded.shape} (expected {expected_w2}), "
                f"b1={b1_loaded.shape} (expected {expected_b1}), "
                f"b2={b2_loaded.shape} (expected {expected_b2}). "
                f"Falling back to random init."
            )
            data.close()
            self._init_random_weights()
            return

        # Cast to float32 to keep inference dtype stable
        self.W1 = w1_loaded.astype(np.float32, copy=False)
        self.b1 = b1_loaded.astype(np.float32, copy=False)
        self.W2 = w2_loaded.astype(np.float32, copy=False)
        self.b2 = b2_loaded.astype(np.float32, copy=False)
        self._update_count = int(_get("update_count", 0))
        self._total_samples = int(_get("total_samples", 0))
        self._cumulative_loss = float(_get("cumulative_loss", 0.0))
        self._version = int(_get("version", 0))

        # Restore learning rate if saved
        if "learning_rate" in data.files:
            self.lr = float(data["learning_rate"])

        # Restore momentum if saved
        if "momentum_W1" in data.files:
            self._momentum_W1 = data["momentum_W1"].astype(np.float32, copy=False)
            self._momentum_b1 = data["momentum_b1"].astype(np.float32, copy=False)
            self._momentum_W2 = data["momentum_W2"].astype(np.float32, copy=False)
            self._momentum_b2 = data["momentum_b2"].astype(np.float32, copy=False)
        else:
            self._momentum_W1 = np.zeros_like(self.W1)
            self._momentum_b1 = np.zeros_like(self.b1)
            self._momentum_W2 = np.zeros_like(self.W2)
            self._momentum_b2 = np.zeros_like(self.b2)

        self._weights_mtime = os.path.getmtime(self._weights_path)
        data.close()

    def rollback_to_checkpoint(self, version: int) -> bool:
        """Rollback to a specific checkpoint version."""
        try:
            checkpoint_path = self._weights_path.replace(".npz", f"_v{version}.npz")
            if os.path.exists(checkpoint_path):
                import shutil
                shutil.copy2(checkpoint_path, self._weights_path)
                self._load_weights()
                return True
        except Exception:
            pass
        return False


class LatentRefiner:
    """
    Refines the latent state z based on current results.

    From TRM paper: z encodes "what we've learned about the query so far"
    and gets updated based on the current answer (scores).

    Supports:
    - Per-collection weight persistence (like TinyScorer)
    - Hot-reload from background worker updates
    - Online learning via learn_from_teacher()
    """

    # Class-level configuration (mirrors TinyScorer)
    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    WEIGHTS_RELOAD_INTERVAL = float(os.environ.get("RERANKER_WEIGHTS_RELOAD_INTERVAL", "60"))

    def __init__(self, dim: int = 256, hidden_dim: int = 256, lr: float = 0.001):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.base_lr = lr
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self._weights_mtime = 0.0
        self._last_reload_check = 0.0
        self._weights_loaded = False

        # Training metrics
        self._update_count = 0
        self._version = 0

        # Momentum for SGD
        self._momentum_W1: Optional[np.ndarray] = None
        self._momentum_b1: Optional[np.ndarray] = None
        self._momentum_W2: Optional[np.ndarray] = None
        self._momentum_b2: Optional[np.ndarray] = None

        # Try to load saved weights, otherwise init random
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
                return
            except Exception as e:
                from scripts.logger import get_logger
                get_logger(__name__).warning(f"LatentRefiner: failed to load {self._weights_path}: {e}, using random init")

        self._init_random_weights()

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        """Sanitize collection name to prevent path traversal."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        """Get weights file path for a collection."""
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"refiner_{safe_name}.npz")

    def set_collection(self, collection: str):
        """Set collection and load corresponding weights."""
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def maybe_reload_weights(self):
        """Check if weights file changed and reload if needed (hot reload)."""
        now = time.time()
        if now - self._last_reload_check < self.WEIGHTS_RELOAD_INTERVAL:
            return
        self._last_reload_check = now

        try:
            if os.path.exists(self._weights_path):
                mtime = os.path.getmtime(self._weights_path)
                if mtime > self._weights_mtime:
                    self._load_weights_safe()
        except Exception:
            pass

    def _load_weights_safe(self):
        """Load weights with advisory file locking (prevents partial reads during writes)."""
        import fcntl
        lock_path = self._weights_path + ".lock"
        try:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                try:
                    self._load_weights()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            self._load_weights()

    def _init_random_weights(self):
        """Initialize with random weights using He initialization."""
        rng = np.random.RandomState(43)
        scale = np.float32(np.sqrt(2.0 / (self.dim * 3)))
        self.W1 = rng.randn(self.dim * 3, self.hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        w2_scale = np.float32(np.sqrt(2.0 / self.hidden_dim))
        self.W2 = rng.randn(self.hidden_dim, self.dim).astype(np.float32) * w2_scale
        self.b2 = np.zeros(self.dim, dtype=np.float32)

        # Initialize momentum
        self._momentum_W1 = np.zeros_like(self.W1)
        self._momentum_b1 = np.zeros_like(self.b1)
        self._momentum_W2 = np.zeros_like(self.W2)
        self._momentum_b2 = np.zeros_like(self.b2)

    def _load_weights(self) -> bool:
        """Load trained weights from disk. Returns True on success."""
        from scripts.logger import get_logger
        logger = get_logger(__name__)
        try:
            data = np.load(self._weights_path, allow_pickle=True)

            # Helper to safely get from NpzFile
            def _get(key: str, default):
                return data[key] if key in data.files else default

            # Validate shapes
            w1 = _get("W1", None)
            w2 = _get("W2", None)
            b1 = _get("b1", None)
            b2 = _get("b2", None)

            if w1 is None or w2 is None:
                data.close()
                return False

            expected_w1 = (self.dim * 3, self.hidden_dim)
            expected_w2 = (self.hidden_dim, self.dim)

            if w1.shape != expected_w1 or w2.shape != expected_w2:
                logger.warning(f"LatentRefiner: shape mismatch W1={w1.shape} W2={w2.shape}")
                data.close()
                return False

            self.W1 = w1.astype(np.float32, copy=False)
            self.b1 = b1.astype(np.float32, copy=False) if b1 is not None else np.zeros(self.hidden_dim, dtype=np.float32)
            self.W2 = w2.astype(np.float32, copy=False)
            self.b2 = b2.astype(np.float32, copy=False) if b2 is not None else np.zeros(self.dim, dtype=np.float32)

            # Load training state
            self._update_count = int(_get("update_count", 0))
            self._version = int(_get("version", 0))

            # Initialize momentum if not loaded
            if self._momentum_W1 is None or self._momentum_W1.shape != self.W1.shape:
                self._momentum_W1 = np.zeros_like(self.W1)
                self._momentum_b1 = np.zeros_like(self.b1)
                self._momentum_W2 = np.zeros_like(self.W2)
                self._momentum_b2 = np.zeros_like(self.b2)

            self._weights_loaded = True
            self._weights_mtime = os.path.getmtime(self._weights_path)
            data.close()
            logger.debug(f"LatentRefiner: loaded weights v{self._version} from {self._weights_path}")
            return True
        except Exception as e:
            logger.warning(f"LatentRefiner: failed to load weights: {e}")
            return False

    def refine(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        alpha: float = 0.5  # EMA smoothing factor
    ) -> np.ndarray:
        """
        Refine latent state based on current ranking.

        Uses attention-weighted sum of top documents as "answer summary".
        """
        # Check for hot-reloaded weights
        self.maybe_reload_weights()

        # Softmax attention over scores to get weighted doc representation
        weights = np.exp(scores - scores.max())
        weights = weights / (weights.sum() + 1e-8)
        doc_summary = (weights[:, None] * doc_embs).sum(axis=0)  # (dim,)

        # Concatenate [z, query, doc_summary]
        x = np.concatenate([z, query_emb, doc_summary])  # (dim*3,)

        # 2-layer refinement
        h = np.maximum(0, x @ self.W1 + self.b1)
        z_new = h @ self.W2 + self.b2

        # EMA update (from TRM: stabilizes training)
        z_refined = alpha * z_new + (1 - alpha) * z

        # Normalize to unit sphere
        z_refined = z_refined / (np.linalg.norm(z_refined) + 1e-8)

        return z_refined

    def refine_with_cache(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        alpha: float = 0.5
    ) -> tuple:
        """Refine with cache for backprop. Returns (z_refined, cache)."""
        weights = np.exp(scores - scores.max())
        weights = weights / (weights.sum() + 1e-8)
        doc_summary = (weights[:, None] * doc_embs).sum(axis=0)

        x = np.concatenate([z, query_emb, doc_summary])
        h = np.maximum(0, x @ self.W1 + self.b1)
        z_new = h @ self.W2 + self.b2
        z_refined = alpha * z_new + (1 - alpha) * z
        z_refined = z_refined / (np.linalg.norm(z_refined) + 1e-8)

        cache = {"x": x, "h": h, "z": z, "z_new": z_new, "alpha": alpha, "weights": weights}
        return z_refined, cache

    def learn_from_teacher(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        teacher_z: np.ndarray,
    ) -> float:
        """
        Online learning: update weights so refined z moves toward teacher_z.

        Uses MSE loss between our z_refined and teacher_z (normalized).

        Returns:
            Loss value
        """
        # Forward pass with cache
        z_refined, cache = self.refine_with_cache(z, query_emb, doc_embs, scores)

        # MSE loss: ||z_refined - teacher_z||^2
        diff = z_refined - teacher_z
        loss = float(np.sum(diff ** 2))

        if loss < 1e-8:
            return 0.0

        # Backward pass (gradient of MSE)
        dz_refined = 2.0 * diff  # (dim,)

        # Through normalization (approx - assume near unit norm)
        dz_new = cache["alpha"] * dz_refined

        # Through W2, b2
        dW2 = np.outer(cache["h"], dz_new)
        db2 = dz_new

        # Through ReLU and W1, b1
        dh = dz_new @ self.W2.T
        dh = dh * (cache["h"] > 0).astype(np.float32)
        dW1 = np.outer(cache["x"], dh)
        db1 = dh

        # SGD with momentum
        momentum = 0.9
        if self._momentum_W1 is None:
            self._momentum_W1 = np.zeros_like(self.W1)
            self._momentum_b1 = np.zeros_like(self.b1)
            self._momentum_W2 = np.zeros_like(self.W2)
            self._momentum_b2 = np.zeros_like(self.b2)

        self._momentum_W1 = momentum * self._momentum_W1 - self.lr * dW1
        self._momentum_b1 = momentum * self._momentum_b1 - self.lr * db1
        self._momentum_W2 = momentum * self._momentum_W2 - self.lr * dW2
        self._momentum_b2 = momentum * self._momentum_b2 - self.lr * db2

        self.W1 += self._momentum_W1
        self.b1 += self._momentum_b1
        self.W2 += self._momentum_W2
        self.b2 += self._momentum_b2

        self._update_count += 1
        return loss

    def learn_from_teacher_with_cache(
        self,
        z: np.ndarray,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        scores: np.ndarray,
        teacher_z: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Online learning with cache for VICReg backprop.

        Like learn_from_teacher(), but returns (z, z_refined, cache) for
        batch-level VICReg regularization.

        Returns:
            (loss, z, z_refined, cache)
        """
        # Forward pass with cache
        z_refined, cache = self.refine_with_cache(z, query_emb, doc_embs, scores)

        # MSE loss: ||z_refined - teacher_z||^2
        diff = z_refined - teacher_z
        loss = float(np.sum(diff ** 2))

        if loss >= 1e-8:
            # Backward pass (gradient of MSE)
            dz_refined = 2.0 * diff

            # Through normalization (approx - assume near unit norm)
            dz_new = cache["alpha"] * dz_refined

            # Through W2, b2
            dW2 = np.outer(cache["h"], dz_new)
            db2 = dz_new

            # Through ReLU and W1, b1
            dh = dz_new @ self.W2.T
            dh = dh * (cache["h"] > 0).astype(np.float32)
            dW1 = np.outer(cache["x"], dh)
            db1 = dh

            # SGD with momentum
            momentum = 0.9
            if self._momentum_W1 is None:
                self._momentum_W1 = np.zeros_like(self.W1)
                self._momentum_b1 = np.zeros_like(self.b1)
                self._momentum_W2 = np.zeros_like(self.W2)
                self._momentum_b2 = np.zeros_like(self.b2)

            self._momentum_W1 = momentum * self._momentum_W1 - self.lr * dW1
            self._momentum_b1 = momentum * self._momentum_b1 - self.lr * db1
            self._momentum_W2 = momentum * self._momentum_W2 - self.lr * dW2
            self._momentum_b2 = momentum * self._momentum_b2 - self.lr * db2

            self.W1 += self._momentum_W1
            self.b1 += self._momentum_b1
            self.W2 += self._momentum_W2
            self.b2 += self._momentum_b2

            self._update_count += 1

        return loss, z, z_refined, cache

    def apply_vicreg_gradient(
        self,
        grad_z_refined: np.ndarray,
        cache: Dict[str, Any],
        weight: float = 0.1,
    ):
        """
        Apply VICReg gradient to refiner weights.

        Called after VICReg.forward() computes gradient w.r.t. z_refined.
        Backprops through the refiner network to update W1, b1, W2, b2.

        Args:
            grad_z_refined: (dim,) gradient from VICReg
            cache: Cache from refine_with_cache() or learn_from_teacher_with_cache()
            weight: Scaling factor for the regularization gradient
        """
        # Backprop through z_refined = alpha * z_new + (1-alpha) * z
        # d/dz_new = alpha * grad_z_refined
        dz_new = cache["alpha"] * grad_z_refined * weight

        # Through W2, b2: z_new = h @ W2 + b2
        dW2 = np.outer(cache["h"], dz_new)
        db2 = dz_new

        # Through ReLU: h = max(0, pre_h)
        dh = dz_new @ self.W2.T
        dh = dh * (cache["h"] > 0).astype(np.float32)

        # Through W1, b1: pre_h = x @ W1 + b1
        dW1 = np.outer(cache["x"], dh)
        db1 = dh

        # Apply gradients directly (VICReg uses same LR as main training)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def _save_weights(self, checkpoint: bool = False):
        """Save weights to disk atomically with file locking.

        Mirrors TinyScorer._save_weights: use a base path without the .npz
        extension for np.savez (which appends .npz automatically), then
        atomically rename the resulting file into place.
        """
        import fcntl

        os.makedirs(self.WEIGHTS_DIR, exist_ok=True)
        self._version += 1

        # np.savez automatically adds .npz, so follow the same pattern as
        # TinyScorer: derive a temporary base path and then construct the
        # actual temp file path that np.savez will create.
        tmp_base = self._weights_path.replace(".npz", ".tmp")
        tmp_path = tmp_base + ".npz"
        lock_path = self._weights_path + ".lock"

        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    np.savez(
                        tmp_base,
                        W1=self.W1,
                        b1=self.b1,
                        W2=self.W2,
                        b2=self.b2,
                        update_count=self._update_count,
                        version=self._version,
                        dim=self.dim,
                        hidden_dim=self.hidden_dim,
                    )
                    os.replace(tmp_path, self._weights_path)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise


# ---------------------------------------------------------------------------
# VICReg: Variance-Invariance-Covariance Regularization for Latent Refinement
# ---------------------------------------------------------------------------
#
# Adapted from VICReg (Bardes et al., 2021) for online 3-pass reranking.
#
# For a refiner that produces z' from z, we regularize the RESIDUAL Δz = z' - z.
# This is appropriate for learned ranking because:
#   - The residual represents "what the predictor learned to add"
#   - Prevents representation collapse (all residuals becoming identical)
#   - Encourages decorrelated, informative updates
#
# Three terms:
#   1. VARIANCE: std(Δz) ≈ target per dimension → prevents collapse
#   2. COVARIANCE: cov(Δz_i, Δz_j) ≈ 0 for i≠j → decorrelation
#   3. INVARIANCE: ||Δz||² small → bounded updates (stability)
#
# Loss = λ_var * var_loss + λ_cov * cov_loss + λ_inv * inv_loss
# ---------------------------------------------------------------------------


class VICReg:
    """
    VICReg regularization for refinement residuals.

    Regularizes the refiner's residual (z_refined - z) to have:
    - Unit variance per dimension (prevents collapse)
    - Decorrelated dimensions (prevents redundancy)
    - Bounded magnitude (stable updates)

    Designed for online learning: accumulate residuals across a batch,
    then call forward() once at batch end.

    Reference: VICReg (Bardes et al., 2021)
    """

    def __init__(
        self,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        lambda_inv: float = 0.1,
        var_target: float = 1.0,
    ):
        """
        Args:
            lambda_var: Weight for variance loss (prevent collapse)
            lambda_cov: Weight for covariance loss (decorrelation)
            lambda_inv: Weight for invariance loss (bounded updates)
            var_target: Target std deviation per dimension (default 1.0)
        """
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_inv = lambda_inv
        self.var_target = var_target

    def forward(
        self, z_batch: np.ndarray, z_refined_batch: np.ndarray
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        Compute VICReg loss and gradient w.r.t. z_refined.

        Args:
            z_batch: (N, dim) original latent states
            z_refined_batch: (N, dim) refined latent states

        Returns:
            (total_loss, grad_z_refined, loss_components)
            - grad_z_refined: gradient w.r.t. z_refined_batch (N, dim)
            - loss_components: dict with var_loss, cov_loss, inv_loss
        """
        N, dim = z_batch.shape
        eps = 1e-8

        # Residual = what the refiner added
        residual = z_refined_batch - z_batch  # (N, dim)
        mean_res = residual.mean(axis=0, keepdims=True)  # (1, dim)
        residual_centered = residual - mean_res  # (N, dim)

        # ===== 1. VARIANCE LOSS =====
        # Goal: std(residual) ≈ var_target per dimension
        # Loss: mean(max(0, var_target - std))  [hinge loss]
        std = residual.std(axis=0) + eps  # (dim,)
        var_diff = self.var_target - std  # positive when std too small
        var_loss = float(np.maximum(0, var_diff).mean())

        # Gradient: d(hinge)/d(residual)
        hinge_mask = (var_diff > 0).astype(np.float32)  # (dim,)
        d_var = -hinge_mask[None, :] * residual_centered / (N * std[None, :] * dim)

        # ===== 2. COVARIANCE LOSS =====
        # Goal: off-diagonal covariance ≈ 0
        # Loss: sum(cov_ij^2) / dim for i ≠ j
        cov = (residual_centered.T @ residual_centered) / (N - 1 + eps)  # (dim, dim)
        off_diag_mask = 1.0 - np.eye(dim, dtype=np.float32)
        off_diag = cov * off_diag_mask
        cov_loss = float((off_diag ** 2).sum() / dim)

        # Gradient: d(L)/d(residual)
        d_cov = 4 * residual_centered @ (off_diag * off_diag_mask) / ((N - 1 + eps) * dim)

        # ===== 3. INVARIANCE LOSS =====
        # Goal: residual magnitude bounded
        # Loss: mean(||residual||²)
        inv_loss = float((residual ** 2).mean())

        # Gradient: d(mean(x²))/d(x) = 2x / (N * dim)
        d_inv = 2 * residual / (N * dim)

        # ===== TOTAL =====
        total_loss = (
            self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
            + self.lambda_inv * inv_loss
        )

        # Gradient w.r.t. z_refined (residual = z_refined - z, so d/dz_refined = d/dresidual)
        grad = (
            self.lambda_var * d_var
            + self.lambda_cov * d_cov
            + self.lambda_inv * d_inv
        ).astype(np.float32)

        components = {
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "inv_loss": inv_loss,
        }

        return total_loss, grad, components


class LearnedProjection:
    """
    Learnable linear projection from embedding dim to working dim.

    Replaces fixed random projection with a learnable layer that adapts
    to domain-specific semantics. Key for true "self-learning search".

    Features:
    - Per-collection weights (like TinyScorer/LatentRefiner)
    - Gradient from downstream scorer/refiner backprops through
    - VICReg-compatible: can receive regularization gradients
    - Hot-reload from background worker updates

    Architecture:
        input (768) → linear → normalize → output (256)

    The projection learns which subspace of BGE is most useful for code search.
    """

    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    WEIGHTS_RELOAD_INTERVAL = float(os.environ.get("RERANKER_WEIGHTS_RELOAD_INTERVAL", "60"))

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
        lr: float = 0.0005,  # Lower LR than scorer - projection is more sensitive
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_lr = lr
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")
        self._weights_mtime = 0.0
        self._last_reload_check = 0.0
        self._weights_loaded = False

        # Training metrics
        self._update_count = 0
        self._version = 0

        # Momentum for SGD
        self._momentum_W: Optional[np.ndarray] = None
        self._momentum = 0.9

        # Try to load saved weights, otherwise init random
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
                return
            except Exception as e:
                from scripts.logger import get_logger
                get_logger(__name__).warning(f"LearnedProjection: failed to load {self._weights_path}: {e}")

        self._init_random_weights()

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"projection_{safe_name}.npz")

    def _init_random_weights(self):
        """Initialize with scaled random weights (Xavier-style)."""
        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        rng = np.random.RandomState(44)  # Deterministic init
        self.W = (rng.randn(self.input_dim, self.output_dim) * scale).astype(np.float32)
        self._momentum_W = np.zeros_like(self.W)

    def set_collection(self, collection: str):
        """Set collection and load corresponding weights."""
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def maybe_reload_weights(self):
        """Check if weights file changed and reload if needed."""
        now = time.time()
        if now - self._last_reload_check < self.WEIGHTS_RELOAD_INTERVAL:
            return
        self._last_reload_check = now

        try:
            if os.path.exists(self._weights_path):
                mtime = os.path.getmtime(self._weights_path)
                if mtime > self._weights_mtime:
                    self._load_weights()
        except Exception:
            pass

    def _load_weights(self):
        """Load weights from disk."""
        import fcntl
        lock_path = self._weights_path + ".lock"
        try:
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                data = np.load(self._weights_path)
                self.W = data["W"].astype(np.float32)
                self._version = int(data.get("version", 0))
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            self._weights_mtime = os.path.getmtime(self._weights_path)
            self._weights_loaded = True
            self._momentum_W = np.zeros_like(self.W)
        except Exception as e:
            from scripts.logger import get_logger
            get_logger(__name__).warning(f"LearnedProjection: load failed: {e}")

    def _save_weights(self):
        """Save weights to disk atomically."""
        import fcntl
        os.makedirs(os.path.dirname(self._weights_path) or ".", exist_ok=True)
        lock_path = self._weights_path + ".lock"
        # np.savez adds .npz extension, so use base path without extension for tmp
        base_path = self._weights_path.rsplit(".npz", 1)[0]
        tmp_path = base_path + ".tmp.npz"

        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                np.savez(tmp_path, W=self.W, version=self._version)
                os.replace(tmp_path, self._weights_path)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            self._weights_mtime = os.path.getmtime(self._weights_path)
        except Exception as e:
            from scripts.logger import get_logger
            get_logger(__name__).warning(f"LearnedProjection: save failed: {e}")

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to output dim (normalized).

        Args:
            embeddings: (batch, input_dim) or (input_dim,)

        Returns:
            projected: (batch, output_dim) or (output_dim,), L2-normalized
        """
        squeeze = embeddings.ndim == 1
        if squeeze:
            embeddings = embeddings.reshape(1, -1)

        # Linear projection
        projected = embeddings @ self.W  # (batch, output_dim)

        # L2 normalize
        norms = np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8
        projected = projected / norms

        if squeeze:
            projected = projected[0]

        return projected

    def forward_with_cache(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass with cache for backprop.

        Returns:
            projected: (batch, output_dim), L2-normalized
            cache: dict with inputs for backward pass
        """
        squeeze = embeddings.ndim == 1
        if squeeze:
            embeddings = embeddings.reshape(1, -1)

        # Linear projection
        pre_norm = embeddings @ self.W  # (batch, output_dim)

        # L2 normalize
        norms = np.linalg.norm(pre_norm, axis=-1, keepdims=True) + 1e-8
        projected = pre_norm / norms

        cache = {
            "input": embeddings,
            "pre_norm": pre_norm,
            "norms": norms,
            "projected": projected,
        }

        if squeeze:
            projected = projected[0]

        return projected, cache

    def backward(
        self, grad_output: np.ndarray, cache: Dict[str, Any], weight: float = 1.0
    ):
        """Backprop gradient through projection and update weights.

        Args:
            grad_output: gradient w.r.t. projected output (batch, output_dim) or (output_dim,)
            cache: from forward_with_cache
            weight: gradient scaling factor
        """
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        embeddings = cache["input"]
        norms = cache["norms"]

        batch_size = embeddings.shape[0]

        # Backprop through L2 normalization
        # d/dx (x/||x||) = (I - x*x^T/||x||^2) / ||x||
        # Simplified: grad_pre_norm = (grad_output - projected * (grad_output · projected)) / norms
        projected = cache["projected"]
        dot = np.sum(grad_output * projected, axis=-1, keepdims=True)
        grad_pre_norm = (grad_output - projected * dot) / norms

        # Gradient w.r.t. W: dL/dW = input^T @ grad_pre_norm
        dW = embeddings.T @ grad_pre_norm / batch_size

        # Apply weight and update with momentum SGD
        dW = dW * weight
        self._momentum_W = self._momentum * self._momentum_W + dW
        self.W -= self.lr * self._momentum_W

        self._update_count += 1

        # Periodic save
        if self._update_count % 100 == 0:
            self._version += 1
            self._save_weights()


class LearnedHybridWeights:
    """
    Learns optimal dense vs. lexical balance per-collection.

    The hybrid score is: score = sigmoid(alpha) * dense + (1 - sigmoid(alpha)) * lexical

    Where alpha is learned from teacher feedback:
    - If teacher prefers results that dense ranked higher → increase alpha
    - If teacher prefers results that lexical ranked higher → decrease alpha

    Features:
    - Per-collection weights
    - Online gradient updates from teacher signal
    - Bounded between 0 and 1 via sigmoid
    """

    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")

        # Alpha controls dense weight: dense_w = sigmoid(alpha)
        # Initialize at 0 → sigmoid(0) = 0.5 (equal weighting)
        self.alpha = 0.0

        # Momentum
        self._momentum_alpha = 0.0
        self._momentum = 0.9

        # Metrics
        self._update_count = 0
        self._version = 0

        # Try to load saved weights
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"hybrid_{safe_name}.npz")

    def set_collection(self, collection: str):
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def _load_weights(self):
        import fcntl
        lock_path = self._weights_path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            data = np.load(self._weights_path)
            self.alpha = float(data["alpha"])
            self._version = int(data.get("version", 0))
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_weights(self):
        import fcntl
        os.makedirs(os.path.dirname(self._weights_path) or ".", exist_ok=True)
        lock_path = self._weights_path + ".lock"
        # np.savez adds .npz extension, so use base path for tmp
        base_path = self._weights_path.rsplit(".npz", 1)[0]
        tmp_base = base_path + ".tmp"
        tmp_path = tmp_base + ".npz"  # What np.savez actually writes
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            np.savez(tmp_base, alpha=self.alpha, version=self._version)
            os.replace(tmp_path, self._weights_path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @property
    def dense_weight(self) -> float:
        """Current dense weight (0-1)."""
        return 1.0 / (1.0 + np.exp(-self.alpha))

    @property
    def lexical_weight(self) -> float:
        """Current lexical weight (0-1)."""
        return 1.0 - self.dense_weight

    def blend(
        self, dense_scores: np.ndarray, lexical_scores: np.ndarray
    ) -> np.ndarray:
        """Blend dense and lexical scores with learned weights."""
        w = self.dense_weight
        return w * dense_scores + (1 - w) * lexical_scores

    def learn_from_teacher(
        self,
        dense_scores: np.ndarray,
        lexical_scores: np.ndarray,
        teacher_scores: np.ndarray,
    ):
        """Update alpha based on which modality better matches teacher.

        Gradient: d_loss/d_alpha = (teacher - blended) * (dense - lexical) * sigmoid'(alpha)
        If teacher prefers dense-high docs more than blended → increase alpha
        """
        w = self.dense_weight
        blended = self.blend(dense_scores, lexical_scores)

        # Normalize scores for comparison
        teacher_norm = (teacher_scores - teacher_scores.mean()) / (teacher_scores.std() + 1e-8)
        blended_norm = (blended - blended.mean()) / (blended.std() + 1e-8)
        dense_norm = (dense_scores - dense_scores.mean()) / (dense_scores.std() + 1e-8)
        lexical_norm = (lexical_scores - lexical_scores.mean()) / (lexical_scores.std() + 1e-8)

        # Error: how much blended differs from teacher ranking
        error = teacher_norm - blended_norm  # (n_docs,)

        # Modality difference: positive where dense > lexical
        modality_diff = dense_norm - lexical_norm  # (n_docs,)

        # Gradient: push alpha toward modality that matches teacher better
        # sigmoid'(alpha) = w * (1 - w)
        sigmoid_grad = w * (1 - w)
        grad = (error * modality_diff).mean() * sigmoid_grad

        # Momentum SGD
        self._momentum_alpha = self._momentum * self._momentum_alpha + grad
        self.alpha += self.lr * self._momentum_alpha

        # Clamp to prevent extreme values
        self.alpha = np.clip(self.alpha, -5.0, 5.0)

        self._update_count += 1
        if self._update_count % 50 == 0:
            self._version += 1
            self._save_weights()


class QueryExpander:
    """
    Learns query expansions (synonyms/related terms) from usage patterns.

    Observes which terms co-occur with successful retrievals and builds
    a lightweight term→expansion mapping per-collection.

    Features:
    - Learns from teacher feedback: which doc terms appear in high-scoring results
    - Per-collection expansion vocabulary
    - Confidence-weighted: only expands with high-confidence associations
    - Decay: old associations fade without reinforcement
    """

    WEIGHTS_DIR = os.environ.get("RERANKER_WEIGHTS_DIR", "/tmp/rerank_weights")
    MAX_EXPANSIONS_PER_TERM = 5
    MIN_CONFIDENCE = 0.3
    DECAY_RATE = 0.995  # Per-update decay for old associations

    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self._collection = "default"
        self._weights_path = self._get_weights_path("default")

        # term → {expansion_term: confidence}
        # confidence in [0, 1], higher = stronger association
        self.expansions: Dict[str, Dict[str, float]] = {}

        self._update_count = 0
        self._version = 0

        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    @staticmethod
    def _sanitize_collection(collection: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in collection)

    def _get_weights_path(self, collection: str) -> str:
        safe_name = self._sanitize_collection(collection)
        return os.path.join(self.WEIGHTS_DIR, f"expander_{safe_name}.json")

    def set_collection(self, collection: str):
        self._collection = collection
        self._weights_path = self._get_weights_path(collection)
        if os.path.exists(self._weights_path):
            try:
                self._load_weights()
            except Exception:
                pass

    def _load_weights(self):
        import json
        import fcntl
        lock_path = self._weights_path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            with open(self._weights_path, "r") as f:
                data = json.load(f)
            self.expansions = data.get("expansions", {})
            self._version = data.get("version", 0)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save_weights(self):
        import json
        import fcntl
        os.makedirs(os.path.dirname(self._weights_path) or ".", exist_ok=True)
        lock_path = self._weights_path + ".lock"
        tmp_path = self._weights_path + ".tmp"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            with open(tmp_path, "w") as f:
                json.dump({"expansions": self.expansions, "version": self._version}, f)
            os.replace(tmp_path, self._weights_path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for expansion learning."""
        # Use the same tokenizer as the rest of the system
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())
        # Filter common tokens
        return [t for t in tokens if len(t) > 2 and t not in _COMMON_TOKENS]

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """Return expansion terms for the query.

        Args:
            query: Original query string
            max_expansions: Max terms to add

        Returns:
            List of expansion terms (may be empty)
        """
        query_tokens = set(self._tokenize(query))
        candidates: List[Tuple[str, float]] = []

        for token in query_tokens:
            if token in self.expansions:
                for exp_term, conf in self.expansions[token].items():
                    if exp_term not in query_tokens and conf >= self.MIN_CONFIDENCE:
                        candidates.append((exp_term, conf))

        # Sort by confidence, take top
        candidates.sort(key=lambda x: -x[1])
        return [term for term, _ in candidates[:max_expansions]]

    def learn_from_teacher(
        self,
        query: str,
        doc_texts: List[str],
        teacher_scores: np.ndarray,
    ):
        """Learn term associations from teacher-scored documents.

        High-scoring docs contribute their terms as expansions for query terms.
        """
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return

        # Normalize teacher scores to weights
        weights = np.exp(teacher_scores - teacher_scores.max())
        weights = weights / (weights.sum() + 1e-8)

        # Collect doc terms weighted by teacher score
        doc_term_weights: Dict[str, float] = {}
        for doc_text, weight in zip(doc_texts, weights):
            for token in self._tokenize(doc_text):
                if token not in query_tokens:  # Only non-query terms
                    doc_term_weights[token] = doc_term_weights.get(token, 0.0) + weight

        # Update expansion associations
        for query_term in query_tokens:
            if query_term not in self.expansions:
                self.expansions[query_term] = {}

            term_expansions = self.expansions[query_term]

            # Decay existing associations
            for exp in list(term_expansions.keys()):
                term_expansions[exp] = float(term_expansions[exp] * self.DECAY_RATE)
                if term_expansions[exp] < 0.01:
                    del term_expansions[exp]

            # Reinforce associations from high-scoring docs
            for doc_term, weight in doc_term_weights.items():
                if weight > 0.1:  # Only significant weights
                    old_conf = term_expansions.get(doc_term, 0.0)
                    # EMA update
                    new_conf = old_conf + self.lr * (weight - old_conf)
                    # Convert to native float for JSON serialization
                    term_expansions[doc_term] = float(min(new_conf, 1.0))

            # Prune to max expansions per term
            if len(term_expansions) > self.MAX_EXPANSIONS_PER_TERM * 2:
                sorted_exp = sorted(term_expansions.items(), key=lambda x: -x[1])
                self.expansions[query_term] = dict(sorted_exp[:self.MAX_EXPANSIONS_PER_TERM])

        self._update_count += 1
        if self._update_count % 20 == 0:
            self._version += 1
            self._save_weights()

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about learned expansions."""
        total_terms = len(self.expansions)
        total_expansions = sum(len(v) for v in self.expansions.values())
        avg_expansions = total_expansions / max(total_terms, 1)
        return {
            "terms": total_terms,
            "expansions": total_expansions,
            "avg_per_term": avg_expansions,
            "version": self._version,
        }


class ConfidenceEstimator:
    """
    Estimates confidence to enable early stopping.

    From TRM: Q-learning inspired halting - stop when improvement is minimal.
    Uses patience to avoid stopping on noisy single-step improvements.
    """

    def __init__(self, patience: int = 1, min_improvement: float = 0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self._stable_count = 0  # Track consecutive stable iterations

    def reset(self):
        """Reset state for a new query."""
        self._stable_count = 0

    def should_stop(self, state: RefinementState) -> bool:
        """Check if we should stop refining based on score stability."""
        if len(state.score_history) < 2:
            return False

        # Compare current ranking to previous
        prev_scores = state.score_history[-2]
        curr_scores = state.scores

        # Measure ranking correlation (Kendall's tau approximation)
        prev_order = np.argsort(-prev_scores)
        curr_order = np.argsort(-curr_scores)

        # Check if this iteration is "stable" (minimal change)
        is_stable = False

        # If top-k rankings are identical, consider stable
        k = min(5, len(prev_order))
        if np.array_equal(prev_order[:k], curr_order[:k]):
            is_stable = True

        # Check score improvement
        improvement = np.abs(curr_scores - prev_scores).mean()
        if improvement < self.min_improvement:
            is_stable = True

        # Update stable count and check patience
        if is_stable:
            self._stable_count += 1
            if self._stable_count >= self.patience:
                return True
        else:
            self._stable_count = 0  # Reset on meaningful change

        return False


class RecursiveReranker:
    """
    Main recursive reranking pipeline.

    Implements TRM-style iterative refinement:
    1. Initialize latent state z from query
    2. For each iteration:
       a. Score all candidates using [query, doc, z]
       b. Refine z based on current scores
       c. Check for early stopping
    3. Return final ranking

    Key insight: Multiple passes through tiny networks > one pass through large network
    """

    def __init__(
        self,
        n_iterations: int = 3,
        dim: int = 256,
        hidden_dim: int = 512,
        early_stop: bool = True,
        blend_with_initial: float = 0.3,  # Blend with initial scores
    ):
        self.n_iterations = n_iterations
        self.dim = dim
        self.early_stop = early_stop
        self.blend_with_initial = blend_with_initial

        # Initialize components
        self.scorer = TinyScorer(dim=dim, hidden_dim=hidden_dim)
        self.refiner = LatentRefiner(dim=dim)
        # Note: ConfidenceEstimator created per-rerank call for thread safety

        # Try to use ONNX embedder for document encoding
        self._embedder = None
        self._embedder_lock = threading.Lock()

        # Cached projection matrices: input_dim -> projection_matrix
        self._proj_cache: Dict[int, np.ndarray] = {}
        self._proj_cache_lock = threading.Lock()

    def _get_embedder(self):
        """Lazy load embedder for encoding queries and documents."""
        if self._embedder is not None:
            return self._embedder

        with self._embedder_lock:
            if self._embedder is not None:
                return self._embedder

            try:
                from scripts.embedder import get_embedding_model
                model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                self._embedder = get_embedding_model(model_name)
            except Exception:
                self._embedder = None
            return self._embedder

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings with caching and batch optimization."""
        # Check cache first
        cached_results = []
        texts_to_encode = []
        text_indices = []

        for i, text in enumerate(texts):
            cached = _get_cached_embedding(text)
            if cached is not None:
                cached_results.append((i, cached))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)

        # Encode uncached texts in batch
        new_embeddings = []
        if texts_to_encode:
            embedder = self._get_embedder()
            if embedder is not None:
                try:
                    # Batch encode all uncached texts at once
                    embeddings = list(embedder.embed(texts_to_encode))
                    # Validate count matches to avoid IndexError in reconstruction
                    if len(embeddings) != len(texts_to_encode):
                        raise ValueError(f"Embedder returned {len(embeddings)} embeddings for {len(texts_to_encode)} texts")
                    for text, emb in zip(texts_to_encode, embeddings):
                        emb_arr = np.array(emb, dtype=np.float32)
                        # Project to target dim before caching for consistency
                        if emb_arr.shape[0] != self.dim:
                            emb_arr = self._project_to_dim(emb_arr.reshape(1, -1))[0]
                        _cache_embedding(text, emb_arr)
                        new_embeddings.append(emb_arr)
                except Exception:
                    new_embeddings = []

            # Fallback for any that failed - use sha256-derived seed for determinism
            if not new_embeddings:
                import hashlib
                # Determine target dimension: use cached embedding dim if available, else self.dim
                fallback_dim = self.dim
                if cached_results:
                    fallback_dim = cached_results[0][1].shape[0]
                for text in texts_to_encode:
                    # Derive deterministic seed from sha256 (process-stable)
                    text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                    seed = int.from_bytes(text_hash[:4], "big")
                    rng = np.random.RandomState(seed)
                    vec = rng.randn(fallback_dim).astype(np.float32)
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    _cache_embedding(text, vec)
                    new_embeddings.append(vec)

        # Reconstruct results in original order
        result = [None] * len(texts)
        for i, emb in cached_results:
            result[i] = emb
        for i, idx in enumerate(text_indices):
            result[idx] = new_embeddings[i]

        return np.array(result, dtype=np.float32)

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        """Encode texts to raw embeddings WITHOUT projection (for learner).

        Returns embeddings in the model's native dimension (e.g., 768 for BGE).
        Used by CollectionLearner to learn the projection matrix.
        """
        from scripts.embedder import get_model_dimension
        fallback_dim = get_model_dimension()

        embedder = self._get_embedder()
        if embedder is None:
            # Fallback to random embeddings
            import hashlib
            result = []
            for text in texts:
                text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                seed = int.from_bytes(text_hash[:4], "big")
                rng = np.random.RandomState(seed)
                vec = rng.randn(fallback_dim).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                result.append(vec)
            return np.array(result, dtype=np.float32)

        try:
            embeddings = list(embedder.embed(texts))
            result = []
            for emb in embeddings:
                emb_arr = np.array(emb, dtype=np.float32)
                result.append(emb_arr)
            return np.array(result, dtype=np.float32)
        except Exception:
            # Fallback
            import hashlib
            result = []
            for text in texts:
                text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                seed = int.from_bytes(text_hash[:4], "big")
                rng = np.random.RandomState(seed)
                vec = rng.randn(fallback_dim).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                result.append(vec)
            return np.array(result, dtype=np.float32)

    def _project_to_dim(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to target dimension if needed (cached for efficiency)."""
        if embeddings.shape[-1] == self.dim:
            return embeddings

        input_dim = embeddings.shape[-1]

        # Get or create cached projection matrix (deterministic, reused)
        with self._proj_cache_lock:
            if input_dim not in self._proj_cache:
                # Use local RNG for deterministic, process-stable projection
                rng = np.random.RandomState(44)
                proj_matrix = rng.randn(input_dim, self.dim).astype(np.float32) * np.float32(0.01)
                self._proj_cache[input_dim] = proj_matrix

            proj_matrix = self._proj_cache[input_dim]

        projected = embeddings @ proj_matrix
        # Normalize
        norms = np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-8
        return projected / norms

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        initial_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recursively rerank candidates.

        Args:
            query: Search query string
            candidates: List of candidate dicts with 'code', 'path', 'symbol', etc.
            initial_scores: Optional initial scores from hybrid search

        Returns:
            Reranked candidates with updated scores and refinement metadata
        """
        if not candidates:
            return []

        # Create per-call confidence estimator for thread safety
        confidence = ConfidenceEstimator()

        n_docs = len(candidates)

        # Extract document text for encoding
        doc_texts = []
        for c in candidates:
            text_parts = []
            if c.get("symbol"):
                text_parts.append(str(c["symbol"]))
            if c.get("path"):
                text_parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or c.get("text") or ""
            if code:
                text_parts.append(str(code)[:500])  # Truncate for efficiency
            doc_texts.append(" ".join(text_parts) if text_parts else "empty")

        # Encode query and documents
        query_emb = self._encode([query])[0]  # (emb_dim,)
        doc_embs = self._encode(doc_texts)  # (n_docs, emb_dim)

        # Project to working dimension
        query_emb = self._project_to_dim(query_emb.reshape(1, -1))[0]
        doc_embs = self._project_to_dim(doc_embs)

        # Initialize latent state from query
        z = query_emb.copy()

        # Initialize scores from initial_scores or zeros
        if initial_scores is not None:
            scores = np.array(initial_scores, dtype=np.float32)
        else:
            scores = np.zeros(n_docs, dtype=np.float32)

        # Create refinement state
        state = RefinementState(z=z, scores=scores, iteration=0)
        state.score_history.append(scores.copy())

        # Iterative refinement loop
        for i in range(self.n_iterations):
            state.iteration = i + 1

            # Score using current latent state
            new_scores = self.scorer.forward(query_emb, doc_embs, state.z)

            # Blend with previous scores (residual connection from TRM)
            alpha = 0.5  # Weight for new scores
            state.scores = alpha * new_scores + (1 - alpha) * state.scores
            state.score_history.append(state.scores.copy())

            # Refine latent state based on current ranking
            state.z = self.refiner.refine(
                state.z, query_emb, doc_embs, state.scores
            )

            # Check for early stopping
            if self.early_stop and confidence.should_stop(state):
                break

        # Blend with initial scores if provided
        final_scores = state.scores
        if initial_scores is not None and self.blend_with_initial > 0:
            init_arr = np.array(initial_scores, dtype=np.float32)
            # Normalize both to similar scale (clamp std for numerical stability)
            std = final_scores.std()
            if std > 1e-6:
                final_norm = (final_scores - final_scores.mean()) / std
            else:
                final_norm = final_scores - final_scores.mean()
            std = init_arr.std()
            if std > 1e-6:
                init_norm = (init_arr - init_arr.mean()) / std
            else:
                init_norm = init_arr - init_arr.mean()

            final_scores = (1 - self.blend_with_initial) * final_norm + self.blend_with_initial * init_norm

        # Build reranked results
        ranked_indices = np.argsort(-final_scores)
        reranked = []

        # Filename-query correlation boost
        # Boosts results when 2+ query tokens match filename tokens
        # e.g., "hybrid search" matches "hybrid_search.py"
        fname_boost_factor = float(os.environ.get("FNAME_BOOST", "0.15") or 0.15)

        for rank, idx in enumerate(ranked_indices):
            candidate = candidates[idx].copy()
            candidate["recursive_score"] = float(final_scores[idx])
            candidate["recursive_rank"] = rank
            candidate["recursive_iterations"] = state.iteration

            # Add score trajectory for analysis
            trajectory = [float(h[idx]) for h in state.score_history]
            candidate["score_trajectory"] = trajectory

            # Filename boost
            fname_boost = _compute_fname_boost(query, candidate, fname_boost_factor)

            # Update main score
            candidate["score"] = float(final_scores[idx]) + fname_boost
            if fname_boost > 0:
                candidate["fname_boost"] = fname_boost

            reranked.append(candidate)

        # Re-sort if any boosts were applied
        if fname_boost_factor > 0 and any(c.get("fname_boost", 0) > 0 for c in reranked):
            reranked.sort(key=lambda x: -x["score"])

        return reranked


# Convenience function for integration with existing pipeline
def rerank_recursive(
    query: str,
    candidates: List[Dict[str, Any]],
    n_iterations: int = 3,
    blend_with_initial: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper for recursive reranking.

    Args:
        query: Search query
        candidates: Candidate results from hybrid search
        n_iterations: Number of refinement passes
        blend_with_initial: How much to blend with initial scores (0-1)

    Returns:
        Reranked candidates
    """
    reranker = RecursiveReranker(
        n_iterations=n_iterations,
        blend_with_initial=blend_with_initial,
    )

    # Extract initial scores
    initial_scores = [c.get("score", 0.0) for c in candidates]

    return reranker.rerank(query, candidates, initial_scores)


# In-process reranking for MCP integration
def rerank_recursive_inprocess(
    query: str,
    candidates: List[Dict[str, Any]],
    limit: int = 12,
    n_iterations: int = 3,
) -> List[Dict[str, Any]]:
    """
    In-process recursive reranking for MCP server integration.

    Compatible with existing rerank_in_process signature.
    """
    reranked = rerank_recursive(
        query=query,
        candidates=candidates,
        n_iterations=n_iterations,
        blend_with_initial=0.3,
    )

    return reranked[:limit]


# Per-collection learning rerankers (isolated weights per collection)
_LEARNING_RERANKERS: Dict[str, "RecursiveReranker"] = {}
_LEARNING_RERANKERS_LOCK = threading.Lock()


def _get_learning_reranker(
    n_iterations: int = 3,
    dim: int = 256,
    collection: str = "default",
) -> "RecursiveReranker":
    """Get or create a learning reranker for a specific collection."""
    with _LEARNING_RERANKERS_LOCK:
        if collection not in _LEARNING_RERANKERS:
            reranker = RecursiveReranker(n_iterations=n_iterations, dim=dim)
            # Set collection-specific weights path for scorer and refiner
            reranker.scorer.set_collection(collection)
            reranker.refiner.set_collection(collection)
            _LEARNING_RERANKERS[collection] = reranker
        return _LEARNING_RERANKERS[collection]


def rerank_with_learning(
    query: str,
    candidates: List[Dict[str, Any]],
    limit: int = 12,
    n_iterations: int = 3,
    learn_from_onnx: bool = True,
    collection: str = "default",
) -> List[Dict[str, Any]]:
    """
    Learning-enabled reranking for MCP server integration.

    Uses a persistent reranker with weights loaded per-collection.
    Training events are logged for background processing rather than
    inline learning (keeps hot path fast and deterministic).

    Args:
        query: Search query
        candidates: List of candidate documents with scores
        limit: Maximum results to return
        n_iterations: Number of refinement iterations
        learn_from_onnx: Whether to log events for learning (default True)
        collection: Collection name for weight isolation

    Returns:
        Reranked candidates with scores
    """
    reranker = _get_learning_reranker(n_iterations=n_iterations, collection=collection)
    initial_scores = [c.get("score", 0) for c in candidates]

    # Log training event for background processing (no inline teacher scoring by default)
    # If you really want inline teacher scoring, set RERANK_TEACHER_INLINE=1.
    if learn_from_onnx and candidates:
        teacher_scores = None
        if str(os.environ.get("RERANK_TEACHER_INLINE", "")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                from scripts.rerank_local import rerank_local
            except ImportError:
                try:
                    from rerank_local import rerank_local
                except ImportError:
                    rerank_local = None

            if rerank_local is not None:
                try:
                    pairs = []
                    for c in candidates:
                        doc = c.get("code") or c.get("snippet") or ""
                        if not doc:
                            parts = []
                            if c.get("symbol"):
                                parts.append(str(c["symbol"]))
                            if c.get("path"):
                                parts.append(str(c["path"]))
                            doc = " ".join(parts) if parts else "empty"
                        pairs.append((query, doc[:1000]))
                    teacher_scores = rerank_local(pairs)
                except Exception:
                    teacher_scores = None

        try:
            # Try both import paths for Docker (/app/scripts) and local (scripts/)
            try:
                from rerank_events import log_training_event
            except ImportError:
                from scripts.rerank_events import log_training_event
            log_training_event(
                query=query,
                candidates=candidates,
                initial_scores=initial_scores,
                teacher_scores=(list(teacher_scores) if teacher_scores is not None else None),
                collection=collection,
                metadata={"teacher_inline": bool(teacher_scores is not None)},
            )
        except Exception:
            pass  # Best effort - don't fail the request

    # Inference only (no inline learning)
    reranked = reranker.rerank(query, candidates, initial_scores)
    return reranked[:limit]


class ONNXRecursiveReranker(RecursiveReranker):
    """
    Recursive reranker using ONNX cross-encoder for scoring.

    Combines the power of pre-trained cross-encoders with
    iterative refinement from TRM paper.
    """

    def __init__(
        self,
        n_iterations: int = 3,
        onnx_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(n_iterations=n_iterations, **kwargs)

        self.onnx_path = onnx_path or os.environ.get("RERANKER_ONNX_PATH", "")
        self.tokenizer_path = tokenizer_path or os.environ.get("RERANKER_TOKENIZER_PATH", "")

        self._session = None
        self._tokenizer = None
        self._onnx_lock = threading.Lock()

    def _get_onnx_session(self):
        """Load ONNX session and tokenizer."""
        if self._session is not None:
            return self._session, self._tokenizer

        if not HAS_ONNX or not self.onnx_path or not self.tokenizer_path:
            return None, None

        with self._onnx_lock:
            if self._session is not None:
                return self._session, self._tokenizer

            try:
                tok = Tokenizer.from_file(self.tokenizer_path)
                try:
                    tok.enable_truncation(max_length=512)
                except Exception:
                    pass

                sess = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"]
                )

                self._session, self._tokenizer = sess, tok
            except Exception:
                self._session, self._tokenizer = None, None

            return self._session, self._tokenizer

    def _onnx_score(self, query: str, docs: List[str]) -> Optional[np.ndarray]:
        """Score query-document pairs using ONNX cross-encoder."""
        sess, tok = self._get_onnx_session()

        if sess is None or tok is None:
            # Fall back to parent's tiny scorer
            return None

        try:
            pairs = [(query, doc) for doc in docs]
            enc = tok.encode_batch(pairs)

            input_ids = [e.ids for e in enc]
            attn = [e.attention_mask for e in enc]
            max_len = max((len(ids) for ids in input_ids), default=0)

            if max_len == 0:
                return None

            # Get pad token id from tokenizer if available, else use 0
            pad_id = 0
            try:
                pad_token_id = tok.token_to_id("[PAD]")
                if pad_token_id is not None:
                    pad_id = int(pad_token_id)
            except Exception:
                pad_id = 0

            def pad(seq, pad_val):
                return seq + [pad_val] * (max_len - len(seq))

            input_ids_padded = [pad(s, pad_id) for s in input_ids]
            attn_padded = [pad(s, 0) for s in attn]

            # Convert to numpy arrays with proper dtype (required by many ONNX models)
            input_ids_arr = np.array(input_ids_padded, dtype=np.int64)
            attn_arr = np.array(attn_padded, dtype=np.int64)

            input_names = [i.name for i in sess.get_inputs()]
            feeds = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = input_ids_arr
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attn_arr
            if "token_type_ids" in input_names:
                token_type_arr = np.zeros((len(input_ids_padded), max_len), dtype=np.int64)
                feeds["token_type_ids"] = token_type_arr

            out = sess.run(None, feeds)
            logits = out[0]

            scores = []
            for row in logits:
                try:
                    if hasattr(row, "__len__") and len(row) >= 2:
                        scores.append(float(row[1]))
                    elif hasattr(row, "__len__") and len(row) == 1:
                        scores.append(float(row[0]))
                    else:
                        scores.append(float(row))
                except Exception:
                    scores.append(0.0)

            return np.array(scores, dtype=np.float32)

        except Exception:
            return None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        initial_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recursive reranking with ONNX cross-encoder.

        Uses ONNX for scoring, but applies iterative refinement
        through the latent state to capture cross-document relationships.
        """
        if not candidates:
            return []

        # Create per-call confidence estimator for thread safety
        confidence = ConfidenceEstimator()

        # Build document texts
        doc_texts = []
        for c in candidates:
            parts = []
            if c.get("symbol"):
                parts.append(str(c["symbol"]))
            if c.get("path"):
                parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or c.get("text") or ""
            if code:
                parts.append(str(code)[:400])
            doc_texts.append(" ".join(parts) if parts else "empty")

        # Try ONNX scoring
        onnx_scores = self._onnx_score(query, doc_texts)

        if onnx_scores is None:
            # Fall back to parent implementation
            return super().rerank(query, candidates, initial_scores)

        # Initialize with ONNX scores
        scores = onnx_scores.copy()

        # Encode for latent refinement
        query_emb = self._encode([query])[0]
        doc_embs = self._encode(doc_texts)
        query_emb = self._project_to_dim(query_emb.reshape(1, -1))[0]
        doc_embs = self._project_to_dim(doc_embs)

        # Initialize latent state
        z = query_emb.copy()

        state = RefinementState(z=z, scores=scores, iteration=0)
        state.score_history.append(scores.copy())

        # Iterative refinement (refine latent, re-score with ONNX is too expensive)
        # Instead, we use the latent to re-weight the ONNX scores
        for i in range(self.n_iterations - 1):  # Already did one pass with ONNX
            state.iteration = i + 1

            # Refine latent based on current ranking
            state.z = self.refiner.refine(
                state.z, query_emb, doc_embs, state.scores
            )

            # Use tiny scorer with refined latent to get adjustment
            adjustment = self.scorer.forward(query_emb, doc_embs, state.z)

            # Blend ONNX scores with adjustment (adaptive alpha based on convergence)
            try:
                metrics = self.scorer.get_metrics()
                if metrics.get("converged", False) and metrics.get("avg_loss", 1.0) < 0.3:
                    alpha = 0.5  # Higher weight for well-trained scorer
                elif metrics.get("update_count", 0) > 100:
                    alpha = 0.35  # Moderate weight after some training
                else:
                    alpha = 0.2  # Conservative weight for untrained scorer
            except Exception:
                alpha = 0.2  # Fallback to conservative
            state.scores = (1 - alpha) * state.scores + alpha * adjustment
            state.score_history.append(state.scores.copy())

            if self.early_stop and confidence.should_stop(state):
                break

        # Build results
        final_scores = state.scores
        if initial_scores is not None and self.blend_with_initial > 0:
            init_arr = np.array(initial_scores, dtype=np.float32)
            # Clamp std for numerical stability
            std = final_scores.std()
            if std > 1e-6:
                final_norm = (final_scores - final_scores.mean()) / std
            else:
                final_norm = final_scores - final_scores.mean()
            std = init_arr.std()
            if std > 1e-6:
                init_norm = (init_arr - init_arr.mean()) / std
            else:
                init_norm = init_arr - init_arr.mean()
            final_scores = (1 - self.blend_with_initial) * final_norm + self.blend_with_initial * init_norm

        ranked_indices = np.argsort(-final_scores)
        reranked = []

        for rank, idx in enumerate(ranked_indices):
            candidate = candidates[idx].copy()
            candidate["recursive_score"] = float(final_scores[idx])
            candidate["onnx_score"] = float(onnx_scores[idx])
            candidate["recursive_rank"] = rank
            candidate["recursive_iterations"] = state.iteration + 1
            candidate["score_trajectory"] = [float(h[idx]) for h in state.score_history]
            candidate["score"] = float(final_scores[idx])
            reranked.append(candidate)

        return reranked


# Factory function to get best available reranker
def get_recursive_reranker(n_iterations: int = 3, **kwargs) -> RecursiveReranker:
    """
    Get the best available recursive reranker.

    Returns ONNXRecursiveReranker if ONNX model is available,
    otherwise falls back to TinyScorer-based reranker.
    """
    onnx_path = os.environ.get("RERANKER_ONNX_PATH", "")
    tokenizer_path = os.environ.get("RERANKER_TOKENIZER_PATH", "")

    if HAS_ONNX and onnx_path and tokenizer_path:
        return ONNXRecursiveReranker(n_iterations=n_iterations, **kwargs)
    else:
        return RecursiveReranker(n_iterations=n_iterations, **kwargs)


class SessionAwareReranker:
    """
    Session-aware recursive reranker with latent state carryover.

    From TRM paper: carry forward latent z across queries in a session
    to accumulate understanding of what the user is looking for.

    Features:
    - Maintains per-session latent state
    - EMA blending with new query embeddings
    - Automatic decay for stale sessions
    - Thread-safe session management

    Usage:
        reranker = SessionAwareReranker()

        # First query in session
        results1 = reranker.rerank("search function", candidates1, session_id="user_123")

        # Second query - latent carries forward
        results2 = reranker.rerank("search implementation", candidates2, session_id="user_123")
    """

    def __init__(
        self,
        n_iterations: int = 3,
        dim: int = 256,
        session_decay: float = 0.9,  # EMA decay for session latent
        max_session_age: float = 3600.0,  # Seconds before session expires
        max_sessions: int = 1000,  # Max sessions to keep in memory
    ):
        self.n_iterations = n_iterations
        self.dim = dim
        self.session_decay = session_decay
        self.max_session_age = max_session_age
        self.max_sessions = max_sessions

        # Core reranker
        self.reranker = RecursiveReranker(n_iterations=n_iterations, dim=dim)

        # Session state: {session_id: (latent_z, last_access_time)}
        self._sessions: Dict[str, tuple] = {}
        self._session_lock = threading.Lock()

    def _cleanup_old_sessions(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, (_, last_access) in self._sessions.items()
            if now - last_access > self.max_session_age
        ]
        for sid in expired:
            del self._sessions[sid]

        # If still too many, remove oldest
        if len(self._sessions) > self.max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1][1]  # Sort by last_access
            )
            to_remove = len(self._sessions) - self.max_sessions
            for sid, _ in sorted_sessions[:to_remove]:
                del self._sessions[sid]

    def get_session_latent(self, session_id: str) -> Optional[np.ndarray]:
        """Get latent state for a session, if exists and not expired."""
        with self._session_lock:
            if session_id not in self._sessions:
                return None

            latent, last_access = self._sessions[session_id]
            if time.time() - last_access > self.max_session_age:
                del self._sessions[session_id]
                return None

            return latent

    def update_session_latent(self, session_id: str, new_latent: np.ndarray):
        """Update latent state for a session with EMA blending."""
        with self._session_lock:
            self._cleanup_old_sessions()

            if session_id in self._sessions:
                old_latent, _ = self._sessions[session_id]
                # EMA blend: decay * old + (1-decay) * new
                blended = self.session_decay * old_latent + (1 - self.session_decay) * new_latent
                # Normalize
                blended = blended / (np.linalg.norm(blended) + 1e-8)
                self._sessions[session_id] = (blended, time.time())
            else:
                self._sessions[session_id] = (new_latent.copy(), time.time())

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        initial_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank with session-aware latent carryover.

        If session_id is provided:
        1. Initialize latent from session state (if exists) blended with query
        2. Run recursive refinement
        3. Update session state with final latent

        If no session_id, behaves like standard RecursiveReranker.
        """
        if not candidates:
            return []

        # Get session latent if available
        session_latent = None
        if session_id:
            session_latent = self.get_session_latent(session_id)

        # Encode query
        query_emb = self.reranker._encode([query])[0]
        query_emb = self.reranker._project_to_dim(query_emb.reshape(1, -1))[0]

        # Initialize latent: blend session state with query
        if session_latent is not None:
            initial_z = 0.7 * query_emb + 0.3 * session_latent
            initial_z = initial_z / (np.linalg.norm(initial_z) + 1e-8)
        else:
            initial_z = query_emb.copy()

        # Extract document text and encode
        doc_texts = []
        for c in candidates:
            text_parts = []
            if c.get("symbol"):
                text_parts.append(str(c["symbol"]))
            if c.get("path"):
                text_parts.append(str(c["path"]))
            code = c.get("code") or c.get("snippet") or c.get("text") or ""
            if code:
                text_parts.append(str(code)[:500])
            doc_texts.append(" ".join(text_parts) if text_parts else "empty")

        doc_embs = self.reranker._encode(doc_texts)
        doc_embs = self.reranker._project_to_dim(doc_embs)

        # Get initial scores
        if initial_scores is None:
            initial_scores = [c.get("score", 0.0) for c in candidates]

        # Create refinement state with session-initialized latent
        state = RefinementState(z=initial_z, scores=np.array(initial_scores, dtype=np.float32))
        state.score_history.append(state.scores.copy())

        # Iterative refinement loop
        for i in range(self.n_iterations):
            state.iteration = i + 1

            # Score using current latent state
            new_scores = self.reranker.scorer.forward(query_emb, doc_embs, state.z)

            # Blend with previous scores
            alpha = 0.5
            state.scores = alpha * new_scores + (1 - alpha) * state.scores
            state.score_history.append(state.scores.copy())

            # Refine latent state
            state.z = self.reranker.refiner.refine(state.z, query_emb, doc_embs, state.scores)

            # Check for early stopping
            if self.reranker.early_stop and self.reranker.confidence.should_stop(state):
                break

        # Update session state with final latent
        if session_id:
            self.update_session_latent(session_id, state.z)

        # Build reranked results
        final_scores = state.scores
        if self.reranker.blend_with_initial > 0:
            init_arr = np.array(initial_scores, dtype=np.float32)
            # Clamp std for numerical stability
            std = final_scores.std()
            if std > 1e-6:
                final_norm = (final_scores - final_scores.mean()) / std
            else:
                final_norm = final_scores - final_scores.mean()
            std = init_arr.std()
            if std > 1e-6:
                init_norm = (init_arr - init_arr.mean()) / std
            else:
                init_norm = init_arr - init_arr.mean()
            final_scores = (1 - self.reranker.blend_with_initial) * final_norm + self.reranker.blend_with_initial * init_norm

        ranked_indices = np.argsort(-final_scores)
        reranked = []

        for rank, idx in enumerate(ranked_indices):
            candidate = candidates[idx].copy()
            candidate["recursive_score"] = float(final_scores[idx])
            candidate["recursive_rank"] = rank
            candidate["recursive_iterations"] = state.iteration
            candidate["session_aware"] = session_id is not None
            candidate["score_trajectory"] = [float(h[idx]) for h in state.score_history]
            candidate["score"] = float(final_scores[idx])
            reranked.append(candidate)

        return reranked

    def clear_session(self, session_id: str):
        """Clear latent state for a session."""
        with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        with self._session_lock:
            return len(self._sessions)


# Convenience function for session-aware reranking
def rerank_with_session(
    query: str,
    candidates: List[Dict[str, Any]],
    session_id: str,
    n_iterations: int = 3,
) -> List[Dict[str, Any]]:
    """
    Session-aware reranking (stateless convenience wrapper).

    Note: For efficiency, use SessionAwareReranker directly to maintain
    a single instance across requests.
    """
    reranker = SessionAwareReranker(n_iterations=n_iterations)
    return reranker.rerank(query, candidates, session_id=session_id)


if __name__ == "__main__":
    # Quick test
    print("Testing Recursive Reranker...")

    # Mock candidates
    candidates = [
        {"path": "scripts/search.py", "symbol": "hybrid_search", "code": "def hybrid_search(query): pass", "score": 0.8},
        {"path": "scripts/index.py", "symbol": "index_file", "code": "def index_file(path): pass", "score": 0.6},
        {"path": "tests/test_search.py", "symbol": "test_search", "code": "def test_search(): assert True", "score": 0.9},
    ]

    reranker = RecursiveReranker(n_iterations=3)
    results = reranker.rerank("hybrid search implementation", candidates)

    print("\nReranked results:")
    for r in results:
        print(f"  {r['recursive_rank']+1}. {r['symbol']} (score={r['score']:.3f}, iters={r['recursive_iterations']})")
        print(f"     trajectory: {r['score_trajectory']}")

    print("\nDone!")

