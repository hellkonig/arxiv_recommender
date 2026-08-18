from arxiv_recommender.schemas import Paper
from arxiv_recommender.text_vectorization.text_policy import (
    TitleAbstractTextPolicy,
)


def test_title_abstract_policy_describes_and_builds_default_text() -> None:
    paper = Paper(title="A title", abstract="An abstract")
    policy = TitleAbstractTextPolicy()

    assert policy.name == "title_abstract"
    assert policy.version == "1.0.0"
    assert policy.config == {"separator": " "}
    assert policy.build_text(paper) == "A title An abstract"


def test_title_abstract_policy_records_custom_separator() -> None:
    paper = Paper(title="A title", abstract="An abstract")
    policy = TitleAbstractTextPolicy(separator="\n")

    assert policy.config == {"separator": "\n"}
    assert policy.build_text(paper) == "A title\nAn abstract"
