import unittest
from arxiv_recommender.arxiv_paper_fetcher.utils import (
    build_arxiv_query_params,
    remove_control_characters,
    validate_arxiv_category,
    validate_arxiv_date,
)


class TestUtils(unittest.TestCase):
    def test_build_arxiv_query_params(self) -> None:
        """Test building arXiv API query params with category and max_results."""
        result = build_arxiv_query_params(date="20231001", category="cs.AI", max_results=100)
        self.assertEqual(
            result,
            {
                "search_query": "cat:cs.AI AND submittedDate:[202310010000 TO 202310012359]",
                "max_results": 100,
            },
        )

    def test_build_arxiv_query_params_without_category(self) -> None:
        """Test building arXiv API query params with date only."""
        result = build_arxiv_query_params(date="20231001", max_results=100)
        self.assertEqual(
            result,
            {
                "search_query": "submittedDate:[202310010000 TO 202310012359]",
                "max_results": 100,
            },
        )

    def test_build_arxiv_query_params_rejects_invalid_date(self) -> None:
        """Test invalid dates are rejected."""
        invalid_dates = ["", "2023-10-01"]
        for date in invalid_dates:
            with self.subTest(date=date):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"Date must be in YYYYMMDD format; got {date!r}",
                ):
                    build_arxiv_query_params(date=date, category="cs.AI")

    def test_build_arxiv_query_params_rejects_invalid_category(self) -> None:
        """Test invalid categories are rejected."""
        with self.assertRaisesRegex(ValueError, "Category must be an arXiv category"):
            build_arxiv_query_params(date="20231001", category="cs AI")

    def test_validate_arxiv_date_returns_valid_date(self) -> None:
        """Test valid dates are returned unchanged."""
        self.assertEqual(validate_arxiv_date("20231001"), "20231001")

    def test_validate_arxiv_date_rejects_nonexistent_date(self) -> None:
        """Test nonexistent calendar dates are rejected."""
        with self.assertRaisesRegex(
            ValueError,
            "Date must be in YYYYMMDD format; got '20230230'",
        ):
            validate_arxiv_date("20230230")

    def test_validate_arxiv_category_returns_valid_category(self) -> None:
        """Test valid categories are returned unchanged."""
        self.assertEqual(validate_arxiv_category("cs.LG"), "cs.LG")

    def test_validate_arxiv_category_rejects_invalid_categories(self) -> None:
        """Test invalid categories are rejected."""
        invalid_categories = ["", "cs AI", "cat:cs.AI", "cs.AI&max_results=100"]
        for category in invalid_categories:
            with self.subTest(category=category):
                with self.assertRaisesRegex(ValueError, "Category must be an arXiv category"):
                    validate_arxiv_category(category)

    def test_remove_control_characters(self) -> None:
        """Test removing control characters from a string"""
        input_text = "Hello\nWorld\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "Hello World")

        input_text = "Hello\n World\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "Hello World")

        input_text = "\t Hello\n World\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "Hello World")


if __name__ == "__main__":
    unittest.main()
