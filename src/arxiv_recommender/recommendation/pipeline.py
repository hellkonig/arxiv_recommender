from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.favorite_papers import FavoritePapersProvider
from arxiv_recommender.recommendation.recommendation import Recommender
from arxiv_recommender.recommendation.types import RecommendationRunResult
from arxiv_recommender.schemas import AppConfig
from arxiv_recommender.text_vectorization import TextEmbedder
from arxiv_recommender.utils import MetricsCollector


class RecommendationPipeline:
    """Reusable orchestration layer for recommendation runs."""

    def __init__(
        self,
        config: AppConfig,
        fetcher: ArxivFetcher,
        vectorizer: TextEmbedder,
        metrics: MetricsCollector,
        favorite_papers_provider: FavoritePapersProvider,
    ) -> None:
        """Initialize pipeline dependencies."""
        self._config = config
        self._fetcher = fetcher
        self._vectorizer = vectorizer
        self._metrics = metrics
        self._favorite_papers_provider = favorite_papers_provider

    def run(self, date_of_pulling_papers: str | None = None) -> RecommendationRunResult:
        """Execute the full recommendation workflow."""
        favorite_papers = self._favorite_papers_provider.get_papers()

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
