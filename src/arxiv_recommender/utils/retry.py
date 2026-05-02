import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

import requests

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (requests.RequestException,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts (default: 3).
        initial_delay (float): Initial delay in seconds (default: 1.0).
        backoff_factor (float): Multiplier for delay after each retry (default: 2.0).
        exceptions (tuple): Tuple of exception types to catch and retry (default: requests.RequestException).

    Returns:
        Callable: Decorated function that retries on failure.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
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
                        logging.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logging.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry_with_backoff")

        return wrapper

    return decorator
