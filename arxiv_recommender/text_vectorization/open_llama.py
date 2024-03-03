import torch
from transformers import LlamaTokenizer, LlamaModel

class LlamaEmbedding:
    """
    A class using open llama for text vectorization

    Attributes:
        model_path (str): the full path for the pre-trained 
            open llama model
    """
    def __init__(self, model_path):
        self.model_path = model_path

    def tokenize(self, text):
        """
        tokenize the input text

        Args:
            text (src): input text string

        Returns:
            tokens (dict): the keys are 'input_ids' and 'attention_mask',
                and the values are tensors.
        """
        tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        return tokenizer(text, return_tensor='pt')

    def vectorize(self, tokens):
        """
        Get the embedding vectos for the input tokens
        Args:
            token (dict): the output from the self.tokenize

        Returns:
            embedding vectors (tensor): the embedding vectors for each word 
                in the input text string.
        """
        llama_model = LlamaModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        with torch.no_grad():
            outputs = llama_model(**tokens)
        return outputs.last_hidden_stats