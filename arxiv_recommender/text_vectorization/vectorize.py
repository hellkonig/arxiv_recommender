class TextVectorization:
    """
    A class vectorize the input text

    Attributes:
        processor (object): an object includes tokenization 
            and vectorization methods.
    """
    def __init__(self, processor) -> None:
        self.processor = processor
        
    def process(self, text):
        """
        tokenize and vectorize the input text

        Args:
            text (str): input text string

        Returns:
            array: the embedding vector of the input text
        """
        token = self.processor.tokenize(text)
        vector = self.processor.vectorize(token)
        return vector