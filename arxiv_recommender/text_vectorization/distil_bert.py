import torch
from transformers import DistilBertTokenizer, DistilBertModel


class DistilBERTEmbedding:
    """
    A class using DistilBERT for text vectorization

    Attributes:
        model_name (str): the name of the pre-trained DistilBERT model
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name

    def tokenize(self, text):
        """
        Tokenize the input text

        Args:
            text (str): input text string

        Returns:
            tokens: <class 'transformers.tokenization_utils_base.BatchEncoding'>
                the keys are 'input_ids' and 'attention_mask',
                and the values are tensors.
        """
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        return tokenizer(text, return_tensors='pt')

    def vectorize(self, tokens):
        """
        Get the embedding vectors for the input tokens
        Args:
            tokens (dict): the output from the self.tokenize

        Returns:
            embedding vectors (tensor): the embedding vectors for each word 
                in the input text string.
        """
        distilbert_model = DistilBertModel.from_pretrained(self.model_name)
        with torch.no_grad():
            outputs = distilbert_model(**tokens)
        return outputs.last_hidden_state
