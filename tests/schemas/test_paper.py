from datetime import datetime

import pytest

from arxiv_recommender.schemas.paper import Paper


class TestPaper:
    """Tests for Paper schema."""

    def test_create_paper_with_valid_data(self) -> None:
        """Test creating a Paper with valid data."""
        paper = Paper(
            title="Sample Paper Title", abstract="This is a sample abstract for testing purposes."
        )
        assert paper.title == "Sample Paper Title"
        assert paper.abstract == "This is a sample abstract for testing purposes."

    def test_paper_is_frozen(self) -> None:
        """Test that Paper is immutable after creation."""
        paper = Paper(title="Test", abstract="Test abstract")
        with pytest.raises(Exception):
            paper.title = "New Title"

    def test_paper_serialization(self) -> None:
        """Test Paper can be serialized to dict."""
        paper = Paper(title="Test", abstract="Test abstract")
        data = paper.model_dump()
        assert data == {
            "arxiv_id": None,
            "url": None,
            "title": "Test",
            "abstract": "Test abstract",
            "authors": [],
            "categories": [],
            "published": None,
            "updated": None,
        }

    def test_paper_json_serialization(self) -> None:
        """Test Paper can be serialized to JSON."""
        paper = Paper(title="Test", abstract="Test abstract")
        json_str = paper.model_dump_json()
        assert '"title":"Test"' in json_str

    def test_paper_from_dict(self) -> None:
        """Test creating Paper from dictionary."""
        data = {"title": "From Dict", "abstract": "From dict abstract"}
        paper = Paper.model_validate(data)
        assert paper.title == "From Dict"

    def test_paper_required_fields(self) -> None:
        """Test that title and abstract are required."""
        with pytest.raises(Exception):
            Paper()  # type: ignore[call-arg]

    def test_create_paper_with_arxiv_metadata(self) -> None:
        """Test creating a paper with traceable arXiv metadata."""
        paper = Paper(
            arxiv_id="1234.56789",
            url="http://arxiv.org/abs/1234.56789",
            title="Metadata Paper",
            abstract="Metadata abstract",
            authors=["Ada Lovelace", "Grace Hopper"],
            categories=["cs.AI", "cs.LG"],
            published=datetime.fromisoformat("2026-05-01T12:00:00+00:00"),
            updated=datetime.fromisoformat("2026-05-02T12:00:00+00:00"),
        )

        assert paper.arxiv_id == "1234.56789"
        assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
        assert paper.categories == ["cs.AI", "cs.LG"]
        assert paper.published == datetime.fromisoformat("2026-05-01T12:00:00+00:00")
