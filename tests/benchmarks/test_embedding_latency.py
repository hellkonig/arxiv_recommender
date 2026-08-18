import argparse
import json
from typing import Any

import numpy as np
import pytest

from arxiv_recommender.provenance import ModelProvenance
from arxiv_recommender.schemas import AppConfig, VectorizerConfig
from arxiv_recommender.text_vectorization.base import TextEmbedder
from arxiv_recommender.text_vectorization.config import AutoSetting, PoolingStrategy
from arxiv_recommender.text_vectorization.text_policy import (
    PaperTextPolicy,
    TitleAbstractTextPolicy,
)
from benchmarks.embedding_latency import (
    benchmark_embeddings,
    benchmark_size,
    build_representative_inputs,
    positive_int,
)


class FakeVectorizer(TextEmbedder):
    def __init__(self) -> None:
        self.processed_texts: list[str] = []

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(name="fake", version="1.0.0")

    @property
    def text_policy(self) -> PaperTextPolicy:
        return TitleAbstractTextPolicy()

    def process(self, text: str) -> np.ndarray:
        self.processed_texts.append(text)
        return np.array([1.0, 0.0])

    def get_cache_stats(self) -> dict[str, Any]:
        return {}


class FakeTimer:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        self.current += 1.0
        return self.current


def make_config() -> AppConfig:
    return AppConfig(
        favorite_papers_path="favorite_papers.json",
        vectorizer=VectorizerConfig(
            module_name="huggingface_embed",
            class_name="HuggingFaceEmbedding",
            model_name="BAAI/bge-small-en-v1.5",
            cache_size=1000,
            pooling_strategy=PoolingStrategy.AUTO,
            normalize_embeddings=AutoSetting.AUTO,
            max_length=512,
        ),
        top_k=10,
        log_level="INFO",
    )


def test_build_representative_inputs_returns_unique_paper_like_texts() -> None:
    inputs = build_representative_inputs(3)

    assert len(inputs) == 3
    assert len(set(inputs)) == 3
    assert all("Title:" in text for text in inputs)
    assert all("Abstract:" in text for text in inputs)


def test_build_representative_inputs_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError, match="count must be greater than 0"):
        build_representative_inputs(0)


def test_positive_int_accepts_positive_values() -> None:
    assert positive_int("3") == 3


def test_positive_int_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="value must be greater than 0"):
        positive_int("0")


def test_benchmark_embeddings_returns_json_serializable_report() -> None:
    fake_vectorizer = FakeVectorizer()

    report = benchmark_embeddings(
        config=make_config(),
        sizes=[2],
        repeats=2,
        vectorizer_factory=lambda _config: fake_vectorizer,
        timer=FakeTimer(),
    )

    assert report["benchmark"] == "embedding_latency"
    assert report["model"]["model_name"] == "BAAI/bge-small-en-v1.5"
    assert report["model"]["cache_size"] == 0
    assert report["model_load_seconds"] == 1.0
    assert report["warmup_seconds"] == 1.0
    assert report["results"] == [
        {
            "paper_count": 2,
            "repeat_count": 2,
            "mean_seconds": 1.0,
            "best_seconds": 1.0,
            "worst_seconds": 1.0,
            "mean_papers_per_second": 2.0,
        }
    ]
    assert len(fake_vectorizer.processed_texts) == 5
    json.dumps(report)


def test_benchmark_size_measures_repeats_with_unique_inputs() -> None:
    fake_vectorizer = FakeVectorizer()

    result = benchmark_size(
        vectorizer=fake_vectorizer,
        paper_count=2,
        repeats=2,
        timer=FakeTimer(),
    )

    assert result == {
        "paper_count": 2,
        "repeat_count": 2,
        "mean_seconds": 1.0,
        "best_seconds": 1.0,
        "worst_seconds": 1.0,
        "mean_papers_per_second": 2.0,
    }
    assert len(fake_vectorizer.processed_texts) == 4
    assert any("Benchmark repeat: 0" in text for text in fake_vectorizer.processed_texts)
    assert any("Benchmark repeat: 1" in text for text in fake_vectorizer.processed_texts)


@pytest.mark.parametrize(
    ("sizes", "repeats", "match"),
    [
        ([], 1, "sizes must not be empty"),
        ([0], 1, "all sizes must be greater than 0"),
        ([1], 0, "repeats must be greater than 0"),
    ],
)
def test_benchmark_embeddings_validates_inputs(
    sizes: list[int],
    repeats: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        benchmark_embeddings(
            config=make_config(),
            sizes=sizes,
            repeats=repeats,
            vectorizer_factory=lambda _config: FakeVectorizer(),
            timer=FakeTimer(),
        )
