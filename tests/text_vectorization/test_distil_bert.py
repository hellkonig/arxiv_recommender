import unittest
import torch

from arxiv_recommender.text_vectorization.distil_bert import DistilBERTEmbedding

class TestDistilBERTEmbedding(unittest.TestCase):
    def setUp(self):
        self.distilbert_embedding = DistilBERTEmbedding()
        self.text = "This is a test sentence."

    def test_tokenize(self):
        tokens = self.distilbert_embedding.tokenize(self.text)
        self.assertIn('input_ids', tokens)
        self.assertIn('attention_mask', tokens)
        self.assertIsInstance(tokens['input_ids'], torch.Tensor)
        self.assertIsInstance(tokens['attention_mask'], torch.Tensor)
        self.assertEqual(tokens['input_ids'].shape[1], 8)  # [CLS] X [SEP]

    def test_vectorize(self):
        tokens = self.distilbert_embedding.tokenize(self.text)
        vectors = self.distilbert_embedding.vectorize(tokens)
        self.assertIsInstance(vectors, torch.Tensor)
        self.assertEqual(vectors.shape[0], 1)  # Batch size
        self.assertEqual(vectors.shape[1], 8)  # [CLS] X [SEP]
        self.assertEqual(vectors.shape[2], 768)  # Embedding size for DistilBERT

if __name__ == '__main__':
    unittest.main()