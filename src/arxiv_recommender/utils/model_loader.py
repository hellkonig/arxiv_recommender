import importlib
import logging
from types import ModuleType
from typing import Any

from arxiv_recommender.text_vectorization import TextEmbedder


class VectorizationModelLoadError(ImportError):
    """Raised when a configured vectorization model cannot be loaded."""


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
        VectorizationModelLoadError: If the configured model cannot be loaded.
    """
    full_module_name = f"arxiv_recommender.text_vectorization.{module_name}"

    try:
        module = importlib.import_module(full_module_name)
    except ModuleNotFoundError as exc:
        if exc.name == full_module_name:
            logging.error("Vectorizer module '%s' was not found.", full_module_name)
            raise VectorizationModelLoadError(
                f"Vectorizer module '{full_module_name}' was not found."
            ) from exc

        logging.error(
            "Vectorizer module '%s' could not be imported because dependency '%s' was not found.",
            full_module_name,
            exc.name,
        )
        raise VectorizationModelLoadError(
            f"Vectorizer module '{full_module_name}' could not be imported because dependency "
            f"'{exc.name}' was not found."
        ) from exc

    model_class = _get_vectorizer_class(module, class_name, full_module_name)

    try:
        model = model_class(model_name, cache_size=cache_size)
    except Exception as exc:
        logging.error(
            "Failed to instantiate vectorizer '%s.%s' with model '%s': %s",
            full_module_name,
            class_name,
            model_name,
            exc,
        )
        raise VectorizationModelLoadError(
            f"Failed to instantiate vectorizer '{full_module_name}.{class_name}' "
            f"with model '{model_name}'."
        ) from exc

    if not isinstance(model, TextEmbedder):
        logging.error(
            "Invalid vectorization model '%s.%s': instance is not a TextEmbedder",
            full_module_name,
            class_name,
        )
        raise VectorizationModelLoadError(
            f"Vectorizer '{full_module_name}.{class_name}' must create a TextEmbedder."
        )

    return model


def _get_vectorizer_class(
    module: ModuleType,
    class_name: str,
    full_module_name: str,
) -> Any:
    model_class = getattr(module, class_name, None)
    if model_class is None:
        logging.error("Vectorizer class '%s' was not found in '%s'.", class_name, full_module_name)
        raise VectorizationModelLoadError(
            f"Vectorizer class '{class_name}' was not found in module '{full_module_name}'."
        )

    if not isinstance(model_class, type) or not issubclass(model_class, TextEmbedder):
        logging.error(
            "Invalid vectorizer class '%s.%s': not a TextEmbedder subclass.",
            full_module_name,
            class_name,
        )
        raise VectorizationModelLoadError(
            f"Vectorizer class '{full_module_name}.{class_name}' must inherit from TextEmbedder."
        )

    return model_class
