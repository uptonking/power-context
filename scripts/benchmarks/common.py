"""
Common types and utilities for Context-Engine benchmarks.

Provides unified reporting format and proper statistical functions.
"""

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Proper Percentile Calculation
# ---------------------------------------------------------------------------
def percentile(values: List[float], p: float) -> float:
    """
    Compute the p-th percentile of a list of values.
    
    Args:
        values: List of numeric values
        p: Percentile to compute (0.0 to 1.0)
    
    Returns:
        The p-th percentile value, or 0.0 if list is empty
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Linear interpolation method (same as numpy default)
    idx = p * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute standard percentiles (p50, p90, p95, p99)."""
    return {
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "p99": round(percentile(values, 0.99), 2),
        "min": round(min(values), 2) if values else 0.0,
        "max": round(max(values), 2) if values else 0.0,
        "mean": round(statistics.mean(values), 2) if values else 0.0,
    }


# ---------------------------------------------------------------------------
# Unified Reporting Format
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    """Result from evaluating a single query."""
    query: str
    latency_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    retrieved_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetadata:
    """Metadata for a benchmark run."""
    benchmark_name: str
    version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """
    Unified benchmark report format.
    
    All benchmarks should emit this format for consistent analysis.
    """
    metadata: BenchmarkMetadata
    per_query: List[QueryResult] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, str]] = field(default_factory=list)
    
    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from per-query results."""
        if not self.per_query:
            return
        
        latencies = [q.latency_ms for q in self.per_query]
        self.aggregates["latency"] = compute_percentiles(latencies)
        self.aggregates["total_queries"] = len(self.per_query)
        
        # Aggregate all numeric metrics
        metric_keys = set()
        for q in self.per_query:
            metric_keys.update(q.metrics.keys())
        
        for key in metric_keys:
            values = [q.metrics.get(key, 0) for q in self.per_query if key in q.metrics]
            if values:
                self.aggregates[key] = {
                    "mean": round(statistics.mean(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": asdict(self.metadata),
            "per_query": [asdict(q) for q in self.per_query],
            "aggregates": self.aggregates,
            "recommendations": self.recommendations,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print(f"BENCHMARK: {self.metadata.benchmark_name}")
        print("=" * 70)
        print(f"Timestamp: {self.metadata.timestamp}")
        print(f"Queries: {self.aggregates.get('total_queries', len(self.per_query))}")
        
        # Print latency stats
        if "latency" in self.aggregates:
            lat = self.aggregates["latency"]
            print("-" * 70)
            print("LATENCY (ms):")
            print(f"  P50: {lat['p50']:<8} P90: {lat['p90']:<8} P99: {lat['p99']}")
            print(f"  Min: {lat['min']:<8} Max: {lat['max']:<8} Mean: {lat['mean']}")
        
        # Print other aggregates
        print("-" * 70)
        print("METRICS:")
        for key, val in self.aggregates.items():
            if key in ("latency", "total_queries"):
                continue
            if isinstance(val, dict) and "mean" in val:
                print(f"  {key}: {val['mean']:.4f} (min={val['min']:.4f}, max={val['max']:.4f})")
            else:
                print(f"  {key}: {val}")
        
        # Print recommendations
        if self.recommendations:
            print("-" * 70)
            print("RECOMMENDATIONS:")
            for rec in self.recommendations:
                priority = rec.get("priority", "").upper()
                action = rec.get("action", "")
                print(f"  [{priority}] {action}")
        
        print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_report(name: str, config: Optional[Dict] = None) -> BenchmarkReport:
    """Create a new benchmark report with metadata."""
    import os
    
    metadata = BenchmarkMetadata(
        benchmark_name=name,
        config=config or {},
        environment={
            "collection": os.environ.get("COLLECTION_NAME", ""),
            "refrag_runtime": os.environ.get("REFRAG_RUNTIME", ""),
        },
    )
    return BenchmarkReport(metadata=metadata)
