import unittest
from unittest.mock import patch
import torch
from transformers import LlamaTokenizer, LlamaModel

from arxiv_recommender.text_vectorization.open_llama import LlamaEmbedding

class TestLlamaEmbedding(unittest.TestCase):
    def setUp(self):
        self.model_path = 'path_to_your_model'
        self.llama_embedding = LlamaEmbedding(self.model_path)
        self.text = "This is a test sentence."

    @patch('transformers.LlamaTokenizer.from_pretrained')
    def test_tokenize(self, mock_from_pretrained):
        mock_from_pretrained.return_value = LlamaTokenizer()
        tokens = self.llama_embedding.tokenize(self.text)
        self.assertIsInstance(tokens, dict)
        self.assertIn('input_ids', tokens)
        self.assertIn('attention_mask', tokens)
        self.assertIsInstance(tokens['input_ids'], torch.Tensor)
        self.assertIsInstance(tokens['attention_mask'], torch.Tensor)

    @patch('transformers.LlamaModel.from_pretrained')
    def test_vectorize(self, mock_from_pretrained):
        mock_from_pretrained.return_value = LlamaModel()
        tokens = self.llama_embedding.tokenize(self.text)
        vectors = self.llama_embedding.vectorize(tokens)
        self.assertIsInstance(vectors, torch.Tensor)

if __name__ == '__main__':
    unittest.main()