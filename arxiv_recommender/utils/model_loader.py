import importlib
import logging
from typing import Type

def load_vectorization_model(model_name: str = "DistilBERTEmbedding") -> Type:
    """
    Dynamically loads a text vectorization model.

    Args:
        model_name (str): Name of the model class to load.

    Returns:
        Type: The loaded model class.

    Raises:
        ImportError: If the model class is not found.
    """
    try:
        module = importlib.import_module("text_vectorization.distill_bert")
        model_class = getattr(module, model_name)
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        logging.error(f"Failed to load vectorization model '{model_name}': {e}")
        raise ImportError(f"Model '{model_name}' not found.")
