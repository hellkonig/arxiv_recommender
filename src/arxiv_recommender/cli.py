import argparse
import logging

from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.arxiv_paper_fetcher.utils import validate_arxiv_date
from arxiv_recommender.favorite_papers import FileFavoritePapersProvider
from arxiv_recommender.recommendation import RecommendationPipeline
from arxiv_recommender.utils import MetricsCollector, setup_logging
from arxiv_recommender.utils.config_loader import load_config
from arxiv_recommender.utils.logging import SUPPORTED_LOG_LEVELS
from arxiv_recommender.utils.model_loader import load_vectorization_model


def _parse_arxiv_date(value: str) -> str:
    """Adapt arXiv date validation for argparse error handling."""
    try:
        return validate_arxiv_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv Paper Recommender System")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file.",
    )
    parser.add_argument(
        "--date_of_pulling_papers",
        type=_parse_arxiv_date,
        default=None,
        help="Date of pulling papers in YYYYMMDD format. If not provided, defaults to yesterday.",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=SUPPORTED_LOG_LEVELS,
        default=None,
        help=f"Override log level ({', '.join(SUPPORTED_LOG_LEVELS)}).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print metrics summary at the end of execution.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    log_level = args.log_level or config.log_level
    setup_logging(level=log_level, json_format=True)
    logger = logging.getLogger(__name__)

    fetcher = ArxivFetcher()
    vectorizer = load_vectorization_model(
        module_name=config.vectorizer.module_name,
        class_name=config.vectorizer.class_name,
        model_name=config.vectorizer.model_name,
        cache_size=config.vectorizer.cache_size,
        vectorizer_options={
            "pooling_strategy": config.vectorizer.pooling_strategy,
            "normalize_embeddings": config.vectorizer.normalize_embeddings,
            "max_length": config.vectorizer.max_length,
        },
    )
    metrics = MetricsCollector()
    favorite_papers_provider = FileFavoritePapersProvider(
        favorite_papers_path=config.favorite_papers_path,
        fetcher=fetcher,
    )
    pipeline = RecommendationPipeline(
        config=config,
        fetcher=fetcher,
        vectorizer=vectorizer,
        metrics=metrics,
        favorite_papers_provider=favorite_papers_provider,
    )
    result = pipeline.run(date_of_pulling_papers=args.date_of_pulling_papers)

    logger.info("Top recommended papers:")
    for i, recommendation in enumerate(result.recommendations, 1):
        paper = recommendation.paper
        logger.info(
            "%d. %s (%s) [%s]",
            i,
            paper.title,
            paper.abstract,
            paper.url or "URL unavailable",
        )

    if args.stats or log_level == "DEBUG":
        logger.info("Metrics summary: %s", result.metrics_summary)


if __name__ == "__main__":
    main()
