"""Metrics collection for observability.

Tracks application-level metrics including API calls, retries,
cache performance, and embedding latency.

Usage:
    from arxiv_recommender.utils.metrics import MetricsCollector

    metrics = MetricsCollector()
    metrics.increment_api_calls()
    metrics.add_embedding_latency(0.125)

    # Pass to components that need it
    recommender = Recommender(vectorizer, papers, metrics)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsCollector:
    """Collects and aggregates application metrics.

    Attributes:
        api_calls: Number of successful API calls made.
        api_errors: Number of failed API calls.
        retry_attempts: Total number of retry attempts across all operations.
        cache_hits: Number of cache hits.
        cache_misses: Number of cache misses.
        embedding_latencies: List of embedding processing times in seconds.
    """

    api_calls: int = 0
    api_errors: int = 0
    retry_attempts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    embedding_latencies: list[float] = field(default_factory=list)

    def increment_api_calls(self) -> None:
        """Increment the successful API call counter."""
        self.api_calls += 1

    def increment_api_errors(self) -> None:
        """Increment the failed API call counter."""
        self.api_errors += 1

    def increment_retry_attempts(self, count: int = 1) -> None:
        """Increment the retry attempt counter.

        Args:
            count: Number of retry attempts to add. Default is 1.
        """
        self.retry_attempts += count

    def update_cache_stats(self, hits: int, misses: int) -> None:
        """Update cache hit/miss counters.

        Args:
            hits: Number of cache hits.
            misses: Number of cache misses.
        """
        self.cache_hits += hits
        self.cache_misses += misses

    def add_embedding_latency(self, latency_seconds: float) -> None:
        """Record an embedding processing latency.

        Args:
            latency_seconds: Time taken to process embedding in seconds.
        """
        self.embedding_latencies.append(latency_seconds)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all collected metrics.

        Returns:
            Dictionary containing metric names and their values.
            Includes computed averages for latency metrics.
        """
        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0.0

        avg_latency = 0.0
        p95_latency = 0.0
        if self.embedding_latencies:
            latencies_array = np.array(self.embedding_latencies)
            avg_latency = float(np.mean(latencies_array))
            p95_latency = float(np.percentile(latencies_array, 95))

        return {
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "retry_attempts": self.retry_attempts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "embedding_count": len(self.embedding_latencies),
            "embedding_avg_latency_sec": round(avg_latency, 3),
            "embedding_p95_latency_sec": round(p95_latency, 3),
        }
