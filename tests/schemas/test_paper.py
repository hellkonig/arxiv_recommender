import pytest
from arxiv_recommender.schemas.paper import Paper


class TestPaper:
    """Tests for Paper schema."""

    def test_create_paper_with_valid_data(self):
        """Test creating a Paper with valid data."""
        paper = Paper(
            title="Sample Paper Title",
            abstract="This is a sample abstract for testing purposes."
        )
        assert paper.title == "Sample Paper Title"
        assert paper.abstract == "This is a sample abstract for testing purposes."

    def test_paper_is_frozen(self):
        """Test that Paper is immutable after creation."""
        paper = Paper(title="Test", abstract="Test abstract")
        with pytest.raises(Exception):
            paper.title = "New Title"

    def test_paper_serialization(self):
        """Test Paper can be serialized to dict."""
        paper = Paper(title="Test", abstract="Test abstract")
        data = paper.model_dump()
        assert data == {"title": "Test", "abstract": "Test abstract"}

    def test_paper_json_serialization(self):
        """Test Paper can be serialized to JSON."""
        paper = Paper(title="Test", abstract="Test abstract")
        json_str = paper.model_dump_json()
        assert '"title":"Test"' in json_str

    def test_paper_from_dict(self):
        """Test creating Paper from dictionary."""
        data = {"title": "From Dict", "abstract": "From dict abstract"}
        paper = Paper(**data)
        assert paper.title == "From Dict"

    def test_paper_required_fields(self):
        """Test that title and abstract are required."""
        with pytest.raises(Exception):
            Paper()