import importlib
import logging
from typing import Type

def load_vectorization_model(
        module_name: str,
        class_name: str,
        model_name: str,
    ) -> Type:
    """
    Dynamically loads a text vectorization model.

    Args:
        module_name (str): Name of the module to load.
        class_name (str): Name of the model class to load.
        model_name (str): Name of the model to load.

    Returns:
        Type: The loaded model class.

    Raises:
        ImportError: If the model class is not found.
    """
    try:
        module = importlib.import_module(
            f"arxiv_recommender.text_vectorization.{module_name}"
        )
        model_class = getattr(module, class_name)
        return model_class(model_name)
    except (ModuleNotFoundError, AttributeError) as e:
        logging.error(
            f"Failed to load vectorization model '{module_name}.{class_name}': {e}"
        )
        raise ImportError(f"Model '{module_name}.{class_name}' not found.")
