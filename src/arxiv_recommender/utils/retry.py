"""Retry decorator with exponential backoff for handling transient failures."""

import time
from functools import wraps
from typing import Any, Callable, TypeVar

import requests

from arxiv_recommender.utils.logging import get_logger

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (requests.RequestException,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorates a function with retry logic and exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).
        initial_delay: Initial delay in seconds (default: 1.0).
        backoff_factor: Multiplier for delay after each retry (default: 2.0).
        exceptions: Tuple of exception types to catch and retry (default: requests.RequestException).

    Returns:
        Decorated function that retries on failure.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                            attempt + 1,
                            max_retries + 1,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "All %d attempts failed. Last error: %s",
                            max_retries + 1,
                            e,
                        )

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry_with_backoff")

        return wrapper

    return decorator
