"""Canonical paper text construction for embedding inputs."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arxiv_recommender.schemas import Paper


class PaperTextPolicy(str, Enum):
    """Versioned policies for constructing text from paper metadata."""

    TITLE_ABSTRACT_SINGLE_SPACE_V1 = "title_abstract_single_space_v1"


ACTIVE_PAPER_TEXT_POLICY = PaperTextPolicy.TITLE_ABSTRACT_SINGLE_SPACE_V1


def paper_to_embedding_text(paper: Paper) -> str:
    """Construct embedding text using the active title-and-abstract policy."""
    return f"{paper.title} {paper.abstract}"
