import unittest

from arxiv_recommender.utils.metrics import MetricsCollector


class TestMetricsCollector(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = MetricsCollector()

    def test_initial_state(self) -> None:
        """Test that metrics start at zero."""
        self.assertEqual(self.metrics.api_calls, 0)
        self.assertEqual(self.metrics.api_errors, 0)
        self.assertEqual(self.metrics.retry_attempts, 0)
        self.assertEqual(self.metrics.cache_hits, 0)
        self.assertEqual(self.metrics.cache_misses, 0)
        self.assertEqual(len(self.metrics.embedding_latencies), 0)

    def test_increment_api_calls(self) -> None:
        """Test incrementing API call counter."""
        self.metrics.increment_api_calls()
        self.assertEqual(self.metrics.api_calls, 1)

        self.metrics.increment_api_calls()
        self.assertEqual(self.metrics.api_calls, 2)

    def test_increment_api_errors(self) -> None:
        """Test incrementing API error counter."""
        self.metrics.increment_api_errors()
        self.assertEqual(self.metrics.api_errors, 1)

    def test_increment_retry_attempts(self) -> None:
        """Test incrementing retry attempts."""
        self.metrics.increment_retry_attempts()
        self.assertEqual(self.metrics.retry_attempts, 1)

        self.metrics.increment_retry_attempts(3)
        self.assertEqual(self.metrics.retry_attempts, 4)

    def test_update_cache_stats(self) -> None:
        """Test updating cache hit/miss counters."""
        self.metrics.update_cache_stats(hits=5, misses=2)
        self.assertEqual(self.metrics.cache_hits, 5)
        self.assertEqual(self.metrics.cache_misses, 2)

    def test_add_embedding_latency(self) -> None:
        """Test adding embedding latency values."""
        self.metrics.add_embedding_latency(0.1)
        self.metrics.add_embedding_latency(0.2)
        self.metrics.add_embedding_latency(0.3)

        self.assertEqual(len(self.metrics.embedding_latencies), 3)

    def test_get_summary_empty(self) -> None:
        """Test summary with no data."""
        summary = self.metrics.get_summary()

        self.assertEqual(summary["api_calls"], 0)
        self.assertEqual(summary["api_errors"], 0)
        self.assertEqual(summary["retry_attempts"], 0)
        self.assertEqual(summary["cache_hits"], 0)
        self.assertEqual(summary["cache_misses"], 0)
        self.assertEqual(summary["cache_hit_rate"], 0.0)
        self.assertEqual(summary["embedding_count"], 0)
        self.assertEqual(summary["embedding_avg_latency_sec"], 0.0)
        self.assertEqual(summary["embedding_p95_latency_sec"], 0.0)

    def test_get_summary_with_data(self) -> None:
        """Test summary with data."""
        self.metrics.increment_api_calls()
        self.metrics.increment_api_errors()
        self.metrics.increment_retry_attempts(2)
        self.metrics.update_cache_stats(hits=8, misses=2)
        self.metrics.add_embedding_latency(0.1)
        self.metrics.add_embedding_latency(0.2)
        self.metrics.add_embedding_latency(0.3)

        summary = self.metrics.get_summary()

        self.assertEqual(summary["api_calls"], 1)
        self.assertEqual(summary["api_errors"], 1)
        self.assertEqual(summary["retry_attempts"], 2)
        self.assertEqual(summary["cache_hits"], 8)
        self.assertEqual(summary["cache_misses"], 2)
        self.assertEqual(summary["cache_hit_rate"], 0.8)
        self.assertEqual(summary["embedding_count"], 3)
        self.assertAlmostEqual(summary["embedding_avg_latency_sec"], 0.2, places=2)

    def test_cache_hit_rate_zero_on_no_cache(self) -> None:
        """Test that cache hit rate is 0 when there are no cache operations."""
        summary = self.metrics.get_summary()
        self.assertEqual(summary["cache_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
