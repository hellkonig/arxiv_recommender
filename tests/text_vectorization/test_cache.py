import unittest
import numpy as np

from arxiv_recommender.text_vectorization.cache import EmbeddingCache


class TestEmbeddingCache(unittest.TestCase):
    def test_cache_miss_returns_none(self):
        cache = EmbeddingCache(max_size=10)
        result = cache.get("some text")
        self.assertIsNone(result)

    def test_cache_hit_returns_embedding(self):
        cache = EmbeddingCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0])
        cache.put("test text", embedding)
        result = cache.get("test text")
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, embedding)

    def test_cache_eviction_lru(self):
        cache = EmbeddingCache(max_size=2)
        cache.put("text1", np.array([1.0]))
        cache.put("text2", np.array([2.0]))
        cache.get("text1")
        cache.put("text3", np.array([3.0]))
        self.assertIsNone(cache.get("text2"))
        self.assertIsNotNone(cache.get("text1"))

    def test_clear_resets_stats(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("text", np.array([1.0]))
        cache.get("text")
        cache.get("text")
        cache.clear()
        self.assertEqual(cache.stats()["hits"], 0)
        self.assertEqual(cache.stats()["misses"], 0)
        self.assertEqual(cache.stats()["size"], 0)

    def test_stats_calculation(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("text1", np.array([1.0]))
        cache.get("text1")
        cache.get("text2")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["max_size"], 10)
        self.assertEqual(stats["hit_rate"], 0.5)

    def test_same_text_returns_hit(self):
        cache = EmbeddingCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0])
        cache.put("test", embedding)
        cache.get("test")
        cache.get("test")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 0)


if __name__ == "__main__":
    unittest.main()