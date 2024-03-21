class MaximumSinglePair:
    """
    A simple ranking by comparing them with a list of like texts (model) with
    the input list of texts. 
    The order of the input texts is determined by the maximum cosine similarity
    of each input texts.

    Attributes:
        model_path (str): model saved path
        model_name (str): saved model name
        vectorizer (object): text vectorization object
    """
    def __init__(self, model_path, model_name, vectorizer) -> None:
        """
        if the model doesn't exist, the class initialization will generate a model.
        """
        pass

    def generate_model(self, model_path, model_name):
        pass