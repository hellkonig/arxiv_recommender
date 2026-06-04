from datetime import datetime

from pydantic import BaseModel, Field


class Paper(BaseModel):
    arxiv_id: str | None = Field(default=None, description="arXiv paper identifier")
    url: str | None = Field(default=None, description="Canonical paper URL")
    title: str = Field(description="Title of the paper")
    abstract: str = Field(description="Abstract of the paper")
    authors: list[str] = Field(default_factory=list, description="Paper authors")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    published: datetime | None = Field(default=None, description="Publication timestamp")
    updated: datetime | None = Field(default=None, description="Last update timestamp")

    model_config = {"frozen": True}
