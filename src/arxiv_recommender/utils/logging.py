"""Logging utilities for structured JSON logging.

This module provides:
- JSONFormatter: Outputs log records as JSON for log aggregation systems
- setup_logging(): Configures application-wide logging
- get_logger(): Helper to get module-level loggers

Usage:
    from arxiv_recommender.utils.logging import setup_logging, get_logger

    setup_logging(level="INFO", json_format=True)
    logger = get_logger(__name__)
    logger.info("Application started")
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON.

    Converts each log record into a JSON object with standard fields
    (timestamp, level, message, etc.) making it easy to parse and index
    in log aggregation systems like ELK, Datadog, or Splunk.

    Example output:
    {
        "timestamp": "2026-05-06T10:30:00+00:00",
        "level": "INFO",
        "logger": "arxiv_recommender.fetcher",
        "message": "Fetching daily papers",
        "module": "fetcher",
        "function": "get_daily_papers",
        "line": 42
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Convert a log record to JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, default=str)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure application-wide logging with a single handler.

    Should be called once at application startup (e.g., in cli.py).
    Sets up the root logger with a handler that outputs to stdout.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Default is INFO.
        json_format: If True, output JSON to stdout (recommended for production).
            If False, output human-readable format (for development).
            Default is True.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: The logger name. Use __name__ from the calling module for proper
            hierarchy (e.g., "arxiv_recommender.fetcher").

    Returns:
        Logger instance configured with application-wide settings.
    """
    return logging.getLogger(name)
