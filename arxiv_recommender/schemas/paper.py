from pydantic import BaseModel, Field


class Paper(BaseModel):
    title: str = Field(description="Title of the paper")
    abstract: str = Field(description="Abstract of the paper")

    model_config = {"frozen": True}