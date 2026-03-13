# AGENTS.md - Agentic Coding Guidelines

This file provides guidelines for agents working on the arxiv_recommender codebase.
These standards reflect top-tier ML engineering practices in the industry.

## Project Overview

`arxiv_recommender` is a content-based recommendation system using DistilBERT embeddings to suggest relevant arXiv papers based on user preferences.

---

## Build, Lint, and Test Commands

### Running Tests

```bash
# Run all tests
python -m pytest

# Run all tests with verbose output
python -m pytest -v

# Run a single test file
python -m pytest tests/recommendation/test_recommender.py

# Run a single test class
python -m pytest tests/recommendation/test_recommender.py::TestRecommender

# Run a single test method
python -m pytest tests/recommendation/test_recommender.py::TestRecommender::test_recommend_by_papers_with_valid_candidates

# Run tests matching a pattern
python -m pytest -k "recommender"

# Run with coverage
python -m pytest --cov=arxiv_recommender --cov-report=term-missing

# Run using unittest
python -m unittest discover -s tests
python -m unittest tests.recommendation.test_recommender
```

### Running the Application

```bash
# Run CLI from source
python3 -m arxiv_recommender.bin.cli --config path/to/config.json

# Run from project root
python cli.py --config arxiv_recommender/data/config.json
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

---

## Top-Tier ML Engineering Standards

### 1. Data Validation with Pydantic (Required)

All data inputs must be validated using Pydantic models. This includes API responses, configuration, and user inputs.

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import date

class Paper(BaseModel):
    """Validated paper metadata."""
    arxiv_id: str = Field(..., pattern=r"^\d{4}\.\d{4,5}$")
    title: str = Field(..., min_length=1, max_length=500)
    abstract: str = Field(..., min_length=10)
    authors: Optional[List[str]] = None
    published: Optional[date] = None

    @validator("abstract")
    def abstract_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Abstract cannot be empty or whitespace")
        return v.strip()

class RecommendationResult(BaseModel):
    """Validated recommendation output."""
    title: str
    abstract: str
    score: float = Field(..., ge=0.0, le=1.0)
    arxiv_id: Optional[str] = None
```

### 2. Configuration Management

Use Pydantic for configuration with environment variable support.

```python
from pydantic import BaseModel
from typing import Optional
import os

class ModelConfig(BaseModel):
    """Model configuration with env var support."""
    module: str = "distil_bert"
    class_name: str = "DistilBERTEmbedding"
    model_name: str = "distilbert-base-uncased"
    cache_dir: Optional[str] = None

    class Config:
        env_prefix = "MODEL_"

class AppConfig(BaseModel):
    """Main application configuration."""
    favorite_papers_path: str = "data/favorite_papers.json"
    vectorizer: ModelConfig = ModelConfig()
    top_k: int = Field(default=10, ge=1, le=100)
    
    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        import json
        with open(path) as f:
            return cls(**json.load(f))
```

### 3. Pipeline Abstraction

ML pipelines should be abstracted for reproducibility and testing.

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class EmbeddingPipeline(ABC):
    """Abstract base class for embedding pipelines."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for efficiency."""
        pass

class RecommendationPipeline:
    """Main recommendation pipeline orchestrator."""

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        cache_dir: Optional[str] = None
    ):
        self.embedding_pipeline = embedding_pipeline
        self.embedding_cache = EmbeddingCache(cache_dir) if cache_dir else None

    def recommend(
        self,
        favorite_papers: List[Dict[str, str]],
        candidate_papers: List[Dict[str, str]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        # Full pipeline implementation
        pass
```

### 4. Embedding Cache

Implement caching to avoid recomputation.

```python
import pickle
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np

class EmbeddingCache:
    """Disk-backed embedding cache with persistence."""

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._get_key(text)
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = self._get_key(text)
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)
```

### 5. Robust Error Handling with Retry Logic

```python
import time
import functools
from typing import Callable, TypeVar, ParamSpec
from requests.exceptions import RequestException

P = ParamSpec("P")
T = TypeVar("T")

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Callable:
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator

class ArxivFetcher:
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        # Implementation
        pass
```

### 6. Observability - Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "metrics"):
            log_data["metrics"] = record.metrics
        return json.dumps(log_data)

# Usage
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(JSONFormatter())]
)
logger = logging.getLogger(__name__)
logger.info("Recommendation generated", extra={"metrics": {"latency_ms": 150}})
```

### 7. Testing Standards

```python
import unittest
from unittest.mock import MagicMock, patch
import pytest

class TestRecommender(unittest.TestCase):
    """Unit tests following industry standards."""

    def setUp(self):
        self.mock_vectorizer = MagicMock(spec=DistilBERTEmbedding)
        self.mock_vectorizer.process.return_value = np.array([1.0, 2.0])
        self.favorite_papers = [
            {"title": "AI", "abstract": "AI research", "arxiv_id": "2301.00001"}
        ]

    def test_recommend_returns_sorted_by_score(self):
        """Verify recommendations are sorted by similarity score."""
        recommendations = self.recommender.recommend_by_papers(self.candidates)
        scores = [r["score"] for r in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))

    @patch("requests.get")
    def test_fetcher_retries_on_failure(self, mock_get):
        """Verify fetcher retries on transient failures."""
        mock_get.side_effect = RequestException("Transient error")
        with self.assertRaises(RequestException):
            self.fetcher.get_paper_by_id("2301.00001")

    def test_empty_candidates_returns_empty(self):
        """Edge case: empty candidate list returns empty list."""
        result = self.recommender.recommend_by_papers([])
        self.assertEqual(result, [])

    def test_favorite_embeddings_computed_once(self):
        """Verify favorite embeddings are cached and not recomputed."""
        self.recommender.recommend_by_papers(self.candidates)
        call_count = self.mock_vectorizer.process.call_count
        # Should equal number of favorites (computed once)
        self.assertEqual(call_count, len(self.favorite_papers))
```

### 8. Code Quality Tools

Add to project root:

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.9"
strict = false
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

---

## Code Style Guidelines

### Imports (ISE Format)

```python
# 1. Standard library
import os
import logging
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

# 2. Third-party packages
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
import requests

# 3. Local application
from arxiv_recommender.utils.json_handler import load_json
from arxiv_recommender.recommendation.recommendation import Recommender
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `Recommender`, `EmbeddingCache` |
| Functions/methods | snake_case | `recommend_by_papers`, `load_json` |
| Variables | snake_case | `favorite_papers`, `candidate_embeddings` |
| Constants | UPPER_SNAKE_CASE | `BASE_DIR`, `MAX_LENGTH` |
| Private methods | _snake_case | `_compute_favorite_embeddings` |
| Config classes | PascalCase + Config | `AppConfig`, `ModelConfig` |

### Type Hints

- Use `Optional[X]` for Python < 3.10 compatibility
- Use `List[X]`, `Dict[K, V]` from typing
- Add return type `-> None` for procedures

```python
def process(self, text: str) -> torch.Tensor:
def recommend_by_papers(
    self,
    candidate_papers: List[Dict[str, str]],
    top_k: Optional[int] = None
) -> List[Dict[str, str]]:
```

### Docstrings (Google Style)

```python
def recommend_by_papers(
    self,
    candidate_papers: List[Dict[str, str]],
    top_k: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Recommends papers based on highest similarity to favorite papers.

    Args:
        candidate_papers: List of candidate papers with title and abstract.
        top_k: Number of top recommendations to return.

    Returns:
        Ranked list of papers with similarity scores.

    Raises:
        ValueError: If candidate papers list is empty.
    """
```

---

## File Structure

```
arxiv_recommender/
├── arxiv_recommender/
│   ├── __init__.py
│   ├── recommendation/
│   │   ├── recommendation.py    # Core logic
│   │   ├── pipeline.py          # Pipeline abstraction
│   │   └── __init__.py
│   ├── text_vectorization/
│   │   ├── distil_bert.py
│   │   ├── cache.py             # Embedding cache
│   │   └── __init__.py
│   ├── arxiv_paper_fetcher/
│   │   ├── fetcher.py           # With retry logic
│   │   ├── parser.py
│   │   └── __init__.py
│   ├── schemas/                 # Pydantic models (NEW)
│   │   ├── paper.py
│   │   ├── config.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── json_handler.py
│   │   ├── model_loader.py
│   │   └── __init__.py
│   └── data/
├── tests/
│   ├── recommendation/
│   ├── text_vectorization/
│   ├── arxiv_paper_fetcher/
│   └── schemas/
├── cli.py
├── pyproject.toml              # Add this
└── requirements.txt
```

---

## Additional Notes

- Run CLI: `python3 -m arxiv_recommender.bin.cli --config path/to/config.json`
- Use mock objects when testing ML models
- All external inputs must be validated with Pydantic
- Add retry logic for external API calls
- Use structured JSON logging for production observability
- Implement caching for expensive operations (embeddings)