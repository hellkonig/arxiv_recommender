import json
import logging
import unittest

from arxiv_recommender.utils.logging import JSONFormatter, get_logger, setup_logging


class TestJSONFormatter(unittest.TestCase):
    def setUp(self) -> None:
        self.formatter = JSONFormatter()

    def test_format_contains_required_fields(self) -> None:
        """Test that JSON output contains all required fields."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)

        self.assertIn("timestamp", parsed)
        self.assertIn("level", parsed)
        self.assertIn("logger", parsed)
        self.assertIn("message", parsed)
        self.assertIn("module", parsed)
        self.assertIn("function", parsed)
        self.assertIn("line", parsed)

    def test_format_message(self) -> None:
        """Test that message is correctly formatted."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)

        self.assertEqual(parsed["message"], "Test message")
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["logger"], "test_logger")


class TestSetupLogging(unittest.TestCase):
    def test_setup_logging_sets_level(self) -> None:
        """Test that setup_logging sets the correct log level on root logger."""
        setup_logging(level="DEBUG", json_format=False)
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)

    def test_setup_logging_json_format(self) -> None:
        """Test that setup_logging works with JSON format."""
        setup_logging(level="INFO", json_format=True)
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)

    def test_setup_logging_suppresses_third_party(self) -> None:
        """Test that third-party loggers are suppressed."""
        setup_logging(level="INFO", json_format=False)
        self.assertTrue(logging.getLogger("urllib3").level >= logging.WARNING)
        self.assertTrue(logging.getLogger("requests").level >= logging.WARNING)


class TestGetLogger(unittest.TestCase):
    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_module")
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_uses_provided_name(self) -> None:
        """Test that get_logger uses the provided name."""
        logger = get_logger("custom_logger_name")
        self.assertEqual(logger.name, "custom_logger_name")


if __name__ == "__main__":
    unittest.main()
