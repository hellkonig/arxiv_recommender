import unittest
from unittest.mock import MagicMock

from arxiv_recommender.text_vectorization.vectorize import TextVectorization

class TestTextVectorize(unittest.TestCase):
    def setUp(self):
        # Create a mock for the processor object
        self.processor_mock = MagicMock()

        # Set up the mock to return specific values when its methods are called
        self.processor_mock.tokenize.return_value = ["tokenized", "text"]
        self.processor_mock.vectorize.return_value = [1, 2, 3, 4]

        # Pass the mock to TextVectorization instead of an actual instance of the processor
        self.text_vectorization = TextVectorization(self.processor_mock)

    def test_process(self):
        text = "input text"
        result = self.text_vectorization.process(text)
        self.assertEqual(result, [1, 2, 3, 4])

        # assert that the mock's methods were called as expected
        self.processor_mock.tokenize.assert_called_once_with(text)
        self.processor_mock.vectorize.assert_called_once_with(["tokenized", "text"])

if __name__ == '__main__':
    unittest.main()