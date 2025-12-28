"""
Pattern Encoder - Convert pattern signatures to dense vectors.

Uses a combination of:
1. MinHash LSH for structural n-grams → 32-dim sparse signature
2. Control flow one-hot encoding → 16-dim 
3. AST path hashing → 16-dim

Total: 64-dim pattern vector suitable for Qdrant storage.

The key property: structurally similar code produces similar vectors,
regardless of variable names or specific API calls.
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import hashlib
import math

from .extractor import PatternSignature


class PatternEncoder:
    """Encode pattern signatures as dense vectors for similarity search."""
    
    # Output dimensions
    NGRAM_DIM = 32      # For structural n-grams (MinHash)
    CONTROL_DIM = 16    # For control flow features
    PATH_DIM = 16       # For AST paths
    TOTAL_DIM = 64      # Total vector dimension
    
    # MinHash parameters
    NUM_HASH_FUNCS = 32
    HASH_PRIMES = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131
    ]
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        
    def encode(self, signature: PatternSignature) -> List[float]:
        """Encode a pattern signature as a 64-dim vector."""
        vector = []
        
        # 1. Encode structural n-grams via MinHash (32-dim)
        ngram_vec = self._encode_ngrams(signature.structural_ngrams)
        vector.extend(ngram_vec)
        
        # 2. Encode control flow features (16-dim)
        cf_vec = self._encode_control_flow(signature.control_flow)
        vector.extend(cf_vec)
        
        # 3. Encode AST paths (16-dim)
        path_vec = self._encode_paths(signature.ast_paths)
        vector.extend(path_vec)
        
        if self.normalize:
            vector = self._l2_normalize(vector)
            
        return vector
    
    def _encode_ngrams(self, ngrams: Counter) -> List[float]:
        """Encode n-grams using MinHash for Jaccard similarity preservation."""
        if not ngrams:
            return [0.0] * self.NGRAM_DIM
            
        # Get set of n-grams (ignore counts for Jaccard)
        ngram_set = set(ngrams.keys())
        
        # MinHash: for each hash function, find min hash of any element
        minhash = []
        for i, prime in enumerate(self.HASH_PRIMES[:self.NUM_HASH_FUNCS]):
            min_h = float('inf')
            for ngram in ngram_set:
                h = self._hash_ngram(ngram, prime, i)
                min_h = min(min_h, h)
            # Normalize to [0, 1] range
            minhash.append((min_h % 1000) / 1000.0 if min_h != float('inf') else 0.0)
            
        return minhash
    
    def _hash_ngram(self, ngram: Tuple, prime: int, seed: int) -> int:
        """Hash an n-gram tuple deterministically."""
        s = str(ngram).encode()
        h = hashlib.md5(s).digest()
        return (int.from_bytes(h[:4], 'big') * prime + seed) % (2**31)
        
    def _encode_control_flow(self, cf: Dict[str, Any]) -> List[float]:
        """Encode control flow features as a 16-dim vector."""
        vec = [0.0] * self.CONTROL_DIM
        
        # Bin 0-3: Loop depth (one-hot for 0, 1, 2, 3+)
        depth = min(cf.get("max_loop_depth", 0), 3)
        vec[depth] = 1.0
        
        # Bin 4-7: Loop count (one-hot for 0, 1, 2, 3+)
        loops = min(cf.get("loop_count", 0), 3)
        vec[4 + loops] = 1.0
        
        # Bin 8-11: Branch count (one-hot for 0, 1, 2, 3+)
        branches = min(cf.get("branch_count", 0), 3)
        vec[8 + branches] = 1.0
        
        # Bin 12: Has try block
        vec[12] = 1.0 if cf.get("try_count", 0) > 0 else 0.0
        
        # Bin 13: Has except
        vec[13] = 1.0 if cf.get("has_except", False) else 0.0
        
        # Bin 14: Has finally
        vec[14] = 1.0 if cf.get("has_finally", False) else 0.0
        
        # Bin 15: Has nested try
        vec[15] = 1.0 if cf.get("try_count", 0) > 1 else 0.0
        
        return vec
    
    def _encode_paths(self, paths: List[Tuple[str, str, str, int]]) -> List[float]:
        """Encode AST paths as a 16-dim vector via feature hashing."""
        vec = [0.0] * self.PATH_DIM
        
        if not paths:
            return vec
            
        total_count = sum(p[3] for p in paths)
        
        for start, path, end, count in paths:
            # Hash path to bucket
            path_str = f"{start}|{path}|{end}"
            bucket = int(hashlib.md5(path_str.encode()).hexdigest()[:4], 16) % self.PATH_DIM
            # Weighted by frequency
            vec[bucket] += count / total_count
            
        return vec
    
    def _l2_normalize(self, vec: List[float]) -> List[float]:
        """L2 normalize a vector."""
        norm = math.sqrt(sum(x*x for x in vec))
        if norm < 1e-10:
            return vec
        return [x / norm for x in vec]
        
    def similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two pattern vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a*a for a in vec_a))
        norm_b = math.sqrt(sum(b*b for b in vec_b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)

