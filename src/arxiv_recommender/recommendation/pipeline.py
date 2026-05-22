from typing import Callable

from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.recommendation.recommendation import Recommender
from arxiv_recommender.recommendation.types import RecommendationRunResult
from arxiv_recommender.schemas import AppConfig, Paper
from arxiv_recommender.text_vectorization import TextEmbedder
from arxiv_recommender.utils import MetricsCollector
from arxiv_recommender.utils.model_loader import load_vectorization_model
from arxiv_recommender.utils.paper_loader import load_favorite_papers
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user


class RecommendationPipeline:
    """Reusable orchestration layer for recommendation runs."""

    def __init__(
        self,
        config: AppConfig,
        fetcher: ArxivFetcher | None = None,
        vectorizer: TextEmbedder | None = None,
        metrics: MetricsCollector | None = None,
        favorite_papers_loader: Callable[[str], list[Paper]] = load_favorite_papers,
        favorite_papers_prompt: Callable[
            [str, ArxivFetcher], list[Paper]
        ] = get_favorite_papers_from_user,
    ) -> None:
        """Initialize pipeline dependencies."""
        self._config = config
        self._fetcher = fetcher or ArxivFetcher()
        self._vectorizer = vectorizer or load_vectorization_model(
            module_name=config.vectorizer.module_name,
            class_name=config.vectorizer.class_name,
            model_name=config.vectorizer.model_name,
            cache_size=config.vectorizer.cache_size,
        )
        self._metrics = metrics or MetricsCollector()
        self._favorite_papers_loader = favorite_papers_loader
        self._favorite_papers_prompt = favorite_papers_prompt

    def run(self, date_of_pulling_papers: str | None = None) -> RecommendationRunResult:
        """Execute the full recommendation workflow."""
        favorite_papers = self._favorite_papers_loader(self._config.favorite_papers_path)
        if not favorite_papers:
            favorite_papers = self._favorite_papers_prompt(
                self._config.favorite_papers_path, self._fetcher
            )

        recommender = Recommender(self._vectorizer, favorite_papers, self._metrics)
        daily_papers = self._fetcher.get_daily_papers(date=date_of_pulling_papers)
        recommendations = recommender.recommend_by_papers(daily_papers, top_k=self._config.top_k)

        summary = self._metrics.get_summary()
        summary["cache"] = self._vectorizer.get_cache_stats()

        return RecommendationRunResult(
            recommendations=recommendations,
            metrics_summary=summary,
            favorite_papers_count=len(favorite_papers),
            candidate_papers_count=len(daily_papers),
        )
