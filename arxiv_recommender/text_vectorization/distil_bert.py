import torch
from transformers import DistilBertTokenizer, DistilBertModel
from typing import List


class DistilBERTEmbedding:
    """
    A class using DistilBERT for text vectorization with chunking
    to handle long sequences.

    Attributes:
        model_name (str): Pre-trained DistilBERT model name.
        tokenizer (DistilBertTokenizer): Tokenizer for DistilBERT.
        model (DistilBertModel): DistilBERT model for embeddings.
        max_length (int): Maximum token length per chunk (default: 512).
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initializes the tokenizer and model.

        Args:
            model_name (str, optional): Pre-trained DistilBERT model name.
                                        Defaults to "distilbert-base-uncased".
        """
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.max_length = 512  # DistilBERT's max input length

    def process(self, text: str) -> torch.Tensor:
        """
        Generates embeddings for a given text by splitting it into chunks
        and aggregating the chunk embeddings.

        Args:
            text (str): Input text string.

        Returns:
            torch.Tensor: The aggregated text embedding.
        """
        tokenized_chunks = self.tokenize(text)
        return self.vectorize(tokenized_chunks)

    def tokenize(self, text: str) -> List[torch.Tensor]:
        """
        Tokenizes input text into chunks that fit within DistilBERT's limits.

        Args:
            text (str): Input text string.

        Returns:
            List[torch.Tensor]: List of tokenized chunks.
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        return [tokens["input_ids"], tokens["attention_mask"]]

    def vectorize(self, tokenized_chunks: List[torch.Tensor]) -> torch.Tensor:
        """
        Get the embedding vectors for the input splitted tokens
        and aggregating the chunk embeddings.

        Args:
            List[torch.Tensor]: List of tokenized chunks.

        Returns:
            torch.Tensor: The aggregated text embedding.
        """
        input_ids = tokenized_chunks[0]
        attention_mask = tokenized_chunks[1]

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # Shape: (num_chunks, seq_len, hidden_dim)

        # Mean pooling over the token dimension to get a single vector per chunk
        sentence_embedding = torch.mean(embeddings, dim=1)

        sentence_embedding = torch.mean(sentence_embedding, dim=0, keepdim=False)

        return sentence_embedding
