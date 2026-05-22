import importlib
import logging

from arxiv_recommender.text_vectorization import TextEmbedder


def load_vectorization_model(
    module_name: str,
    class_name: str,
    model_name: str,
    cache_size: int,
) -> TextEmbedder:
    """
    Dynamically loads a text vectorization model.

    Args:
        module_name (str): Name of the module to load.
        class_name (str): Name of the model class to load.
        model_name (str): Name of the model to load.
        cache_size (int): Maximum number of embeddings to cache.

    Returns:
        Instantiated text embedder.

    Raises:
        ImportError: If the model class is not found.
    """
    try:
        module = importlib.import_module(f"arxiv_recommender.text_vectorization.{module_name}")
        model_class = getattr(module, class_name)
        model = model_class(model_name, cache_size=cache_size)
        if not isinstance(model, TextEmbedder):
            logging.error(
                "Invalid vectorization model '%s.%s': not a TextEmbedder",
                module_name,
                class_name,
            )
            raise ImportError(f"Model '{module_name}.{class_name}' is not a TextEmbedder.")
        return model
    except (ModuleNotFoundError, AttributeError) as e:
        logging.error("Failed to load vectorization model '%s.%s': %s", module_name, class_name, e)
        raise ImportError(f"Model '{module_name}.{class_name}' not found.")
