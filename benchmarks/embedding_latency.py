"""Measure embedding latency for representative title-and-abstract inputs."""

import argparse
import json
import platform
import statistics
import sys
import textwrap
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from arxiv_recommender.schemas import AppConfig
from arxiv_recommender.text_vectorization.base import TextEmbedder
from arxiv_recommender.utils.config_loader import load_config
from arxiv_recommender.utils.model_loader import load_vectorization_model

Timer = Callable[[], float]
VectorizerFactory = Callable[[AppConfig], TextEmbedder]


class BenchmarkHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults while preserving example formatting."""


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed_value


def build_representative_inputs(count: int) -> list[str]:
    """Build deterministic paper-like title-and-abstract inputs."""
    if count <= 0:
        raise ValueError("count must be greater than 0")

    return [
        (
            f"Title: Representation learning for scientific paper recommendation {index}\n"
            "Abstract: We study content-based recommendation for recent arXiv papers using "
            "title and abstract text. The method embeds papers with a transformer encoder, "
            "compares candidates against a user's favorite papers, and ranks results by "
            "semantic similarity. Experiments analyze latency, cache behavior, and ranking "
            "quality for local research discovery workflows."
        )
        for index in range(count)
    ]


def instantiate_vectorizer(config: AppConfig) -> TextEmbedder:
    """Load the configured vectorizer with caching disabled for latency measurement."""
    return load_vectorization_model(
        module_name=config.vectorizer.module_name,
        class_name=config.vectorizer.class_name,
        model_name=config.vectorizer.model_name,
        cache_size=0,
        vectorizer_options={
            "pooling_strategy": config.vectorizer.pooling_strategy,
            "normalize_embeddings": config.vectorizer.normalize_embeddings,
            "max_length": config.vectorizer.max_length,
        },
    )


def benchmark_embeddings(
    config: AppConfig,
    sizes: Sequence[int],
    repeats: int,
    vectorizer_factory: VectorizerFactory = instantiate_vectorizer,
    timer: Timer = perf_counter,
) -> dict[str, Any]:
    """Run the embedding latency benchmark and return a JSON-serializable report."""
    if not sizes:
        raise ValueError("sizes must not be empty")
    if any(size <= 0 for size in sizes):
        raise ValueError("all sizes must be greater than 0")
    if repeats <= 0:
        raise ValueError("repeats must be greater than 0")

    load_start = timer()
    vectorizer = vectorizer_factory(config)
    model_load_seconds = timer() - load_start

    warmup_seconds = run_warmup(vectorizer, timer)
    results = [
        benchmark_size(vectorizer=vectorizer, paper_count=size, repeats=repeats, timer=timer)
        for size in sizes
    ]

    return {
        "benchmark": "embedding_latency",
        "model": model_metadata(config),
        "environment": environment_metadata(),
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "results": results,
    }


def run_warmup(vectorizer: TextEmbedder, timer: Timer) -> float:
    """Run one warmup embedding before measuring benchmark sizes."""
    inputs = build_representative_inputs(1)
    start = timer()
    embed_inputs(vectorizer, inputs)
    return timer() - start


def benchmark_size(
    vectorizer: TextEmbedder,
    paper_count: int,
    repeats: int,
    timer: Timer,
) -> dict[str, Any]:
    """Measure repeated embedding runs for one paper-count size.

    Each repeat embeds a fresh set of deterministic inputs with a repeat marker
    appended. That keeps cache-disabled measurements representative today and
    prevents accidental cache hits if a future vectorizer ignores cache_size=0.
    """
    elapsed_seconds = []
    for repeat_index in range(repeats):
        inputs = build_representative_inputs(paper_count)
        inputs = [f"{text}\nBenchmark repeat: {repeat_index}" for text in inputs]
        start = timer()
        embed_inputs(vectorizer, inputs)
        elapsed_seconds.append(timer() - start)

    mean_seconds = statistics.fmean(elapsed_seconds)
    return {
        "paper_count": paper_count,
        "repeat_count": repeats,
        "mean_seconds": mean_seconds,
        "best_seconds": min(elapsed_seconds),
        "worst_seconds": max(elapsed_seconds),
        "mean_papers_per_second": paper_count / mean_seconds if mean_seconds > 0 else None,
    }


def embed_inputs(vectorizer: TextEmbedder, inputs: Sequence[str]) -> None:
    """Embed every input text."""
    for text in inputs:
        vectorizer.process(text)


def model_metadata(config: AppConfig) -> dict[str, Any]:
    """Return benchmark-relevant model configuration."""
    normalize_embeddings = config.vectorizer.normalize_embeddings
    if hasattr(normalize_embeddings, "value"):
        normalize_embeddings = normalize_embeddings.value

    return {
        "module_name": config.vectorizer.module_name,
        "class_name": config.vectorizer.class_name,
        "model_name": config.vectorizer.model_name,
        "pooling_strategy": config.vectorizer.pooling_strategy.value,
        "normalize_embeddings": normalize_embeddings,
        "max_length": config.vectorizer.max_length,
        "cache_size": 0,
    }


def environment_metadata() -> dict[str, Any]:
    """Return local runtime metadata for interpreting benchmark results."""
    torch_version = None
    torch_cuda_available = None
    if torch is not None:
        torch_version = torch.__version__
        torch_cuda_available = torch.cuda.is_available()

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch_version,
        "torch_cuda_available": torch_cuda_available,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure embedding latency for the configured vectorizer on "
            "representative title-and-abstract inputs."
        ),
        formatter_class=BenchmarkHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              uv run python benchmarks/embedding_latency.py --config configs/config.json
              uv run python benchmarks/embedding_latency.py --config configs/config.json --sizes 1 10 100 --repeats 3

            Argument notes:
              --sizes controls the paper-count batches to time. For example,
                '--sizes 1 10 100' measures one-paper, ten-paper, and
                hundred-paper runs.
              --repeats controls how many measured runs are executed for each
                size. The report includes mean, best, and worst seconds.

            The benchmark runs one warmup embedding after model load. Warmup
            time is reported separately as warmup_seconds and excluded from
            per-size throughput results.
            """
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        default=argparse.SUPPRESS,
        metavar="PATH",
        help=(
            "Application config JSON to load. The benchmark uses its vectorizer "
            "settings and disables the embedding cache for measurement."
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=positive_int,
        default=[1, 10, 100],
        metavar="N",
        help="One or more paper counts to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=3,
        metavar="N",
        help="Measured runs to execute for each size.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the embedding latency benchmark."""
    args = parse_args()
    config = load_config(args.config)
    report = benchmark_embeddings(
        config=config,
        sizes=args.sizes,
        repeats=args.repeats,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
