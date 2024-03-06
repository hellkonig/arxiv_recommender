import unittest
import torch

from arxiv_recommender.text_vectorization.open_llama import LlamaEmbedding

class TestLlamaEmbedding(unittest.TestCase):
    def setUp(self):
        self.model_path = './models/open_llama_3b'
        self.llama_embedding = LlamaEmbedding(self.model_path)
        self.text = "This is a test sentence."

    def test_tokenize(self):
        tokens = self.llama_embedding.tokenize(self.text)
        self.assertIn('input_ids', tokens)
        self.assertIn('attention_mask', tokens)
        self.assertIsInstance(tokens['input_ids'], torch.Tensor)
        self.assertIsInstance(tokens['attention_mask'], torch.Tensor)

    def test_vectorize(self):
        tokens = self.llama_embedding.tokenize(self.text)
        vectors = self.llama_embedding.vectorize(tokens)
        self.assertIsInstance(vectors, torch.Tensor)
        self.assertEqual(vectors.shape[1], 7)

if __name__ == '__main__':
    unittest.main()