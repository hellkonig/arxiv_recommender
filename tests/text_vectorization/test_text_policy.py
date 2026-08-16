from arxiv_recommender.schemas import Paper
from arxiv_recommender.text_vectorization.text_policy import (
    ACTIVE_PAPER_TEXT_POLICY,
    PaperTextPolicy,
    paper_to_embedding_text,
)


def test_active_text_policy_builds_title_then_abstract_with_single_space() -> None:
    paper = Paper(title="A title", abstract="An abstract")

    assert ACTIVE_PAPER_TEXT_POLICY is PaperTextPolicy.TITLE_ABSTRACT_SINGLE_SPACE_V1
    assert paper_to_embedding_text(paper) == "A title An abstract"
