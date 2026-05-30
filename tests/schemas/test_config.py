import pytest
from arxiv_recommender.schemas.config import AppConfig, VectorizerConfig


class TestVectorizerConfig:
    """Tests for VectorizerConfig schema."""

    def test_create_vectorizer_config(self) -> None:
        """Test creating VectorizerConfig with valid data."""
        config = VectorizerConfig(
            module_name="distil_bert",
            class_name="DistilBERTEmbedding",
            model_name="distilbert-base-uncased",
        )
        assert config.module_name == "distil_bert"
        assert config.class_name == "DistilBERTEmbedding"
        assert config.model_name == "distilbert-base-uncased"

    def test_vectorizer_is_frozen(self) -> None:
        """Test that VectorizerConfig is immutable."""
        config = VectorizerConfig(
            module_name="test", class_name="TestClass", model_name="test-model"
        )
        with pytest.raises(Exception):
            config.module_name = "new_module"

    def test_vectorizer_serialization(self) -> None:
        """Test VectorizerConfig can be serialized to dict."""
        config = VectorizerConfig(
            module_name="test", class_name="TestClass", model_name="test-model"
        )
        data = config.model_dump()
        assert data["module_name"] == "test"
        assert data["class_name"] == "TestClass"
        assert data["model_name"] == "test-model"

    def test_vectorizer_rejects_negative_cache_size(self) -> None:
        """Test that cache_size cannot be negative."""
        with pytest.raises(Exception):
            VectorizerConfig(
                module_name="test",
                class_name="TestClass",
                model_name="test-model",
                cache_size=-1,
            )

    def test_vectorizer_allows_zero_cache_size(self) -> None:
        """Test that cache_size can be zero to disable caching."""
        config = VectorizerConfig(
            module_name="test",
            class_name="TestClass",
            model_name="test-model",
            cache_size=0,
        )
        assert config.cache_size == 0


class TestAppConfig:
    """Tests for AppConfig schema."""

    def test_create_app_config(self) -> None:
        """Test creating AppConfig with valid data."""
        config = AppConfig(
            favorite_papers_path="favorite_papers.json",
            vectorizer=VectorizerConfig(
                module_name="distil_bert",
                class_name="DistilBERTEmbedding",
                model_name="distilbert-base-uncased",
            ),
            top_k=10,
        )
        assert config.favorite_papers_path == "favorite_papers.json"
        assert config.top_k == 10
        assert config.vectorizer.module_name == "distil_bert"

    def test_app_config_default_top_k(self) -> None:
        """Test that top_k defaults to 10."""
        config = AppConfig(
            favorite_papers_path="favorites.json",
            vectorizer=VectorizerConfig(
                module_name="test", class_name="TestClass", model_name="test"
            ),
        )
        assert config.top_k == 10

    def test_app_config_custom_top_k(self) -> None:
        """Test custom top_k value."""
        config = AppConfig(
            favorite_papers_path="favorites.json",
            vectorizer=VectorizerConfig(
                module_name="test", class_name="TestClass", model_name="test"
            ),
            top_k=5,
        )
        assert config.top_k == 5

    def test_app_config_is_frozen(self) -> None:
        """Test that AppConfig is immutable."""
        config = AppConfig(
            favorite_papers_path="favorites.json",
            vectorizer=VectorizerConfig(
                module_name="test", class_name="TestClass", model_name="test"
            ),
        )
        with pytest.raises(Exception):
            config.top_k = 20

    def test_app_config_serialization(self) -> None:
        """Test AppConfig can be serialized to dict."""
        config = AppConfig(
            favorite_papers_path="favorites.json",
            vectorizer=VectorizerConfig(
                module_name="test", class_name="TestClass", model_name="test"
            ),
            top_k=5,
        )
        data = config.model_dump()
        assert data["favorite_papers_path"] == "favorites.json"
        assert data["top_k"] == 5
        assert data["vectorizer"]["module_name"] == "test"

    def test_app_config_from_json_dict(self) -> None:
        """Test creating AppConfig from JSON-sourced dict."""
        data = {
            "favorite_papers_path": "my_favorites.json",
            "vectorizer": {
                "module_name": "bert",
                "class_name": "BERTEmbedding",
                "model_name": "bert-base",
            },
            "top_k": 20,
        }
        config = AppConfig.model_validate(data)
        assert config.favorite_papers_path == "my_favorites.json"
        assert config.top_k == 20
        assert config.vectorizer.model_name == "bert-base"

    def test_app_config_required_fields(self) -> None:
        """Test that required fields are enforced."""
        with pytest.raises(Exception):
            AppConfig.model_validate({})

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_app_config_rejects_non_positive_top_k(self, top_k: int) -> None:
        """Test that top_k must be positive."""
        with pytest.raises(Exception):
            AppConfig(
                favorite_papers_path="favorites.json",
                vectorizer=VectorizerConfig(
                    module_name="test", class_name="TestClass", model_name="test"
                ),
                top_k=top_k,
            )

    def test_app_config_rejects_invalid_log_level(self) -> None:
        """Test that log_level must be a supported logging level."""
        with pytest.raises(Exception):
            AppConfig(
                favorite_papers_path="favorites.json",
                vectorizer=VectorizerConfig(
                    module_name="test", class_name="TestClass", model_name="test"
                ),
                log_level="LOUD",
            )

    def test_app_config_normalizes_log_level(self) -> None:
        """Test that lowercase log_level values are normalized."""
        config = AppConfig(
            favorite_papers_path="favorites.json",
            vectorizer=VectorizerConfig(
                module_name="test", class_name="TestClass", model_name="test"
            ),
            log_level="debug",
        )
        assert config.log_level == "DEBUG"
