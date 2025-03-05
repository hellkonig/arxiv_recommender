import json
import logging
from typing import Any

def load_json(file_path: str) -> Any:
    """
    Loads a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: The JSON data, typically a list or dictionary.

    Raises:
        IOError: If the file cannot be read.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        raise

def save_json(file_path: str, data: Any) -> None:
    """
    Saves data to a JSON file.

    Args:
        file_path (str): Path to save the JSON file.
        data (Any): The data to be stored in JSON format.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        logging.info(f"Saved data to {file_path}")
    except IOError as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        raise
