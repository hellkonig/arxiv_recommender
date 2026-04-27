from pydantic import BaseModel, Field


class VectorizerConfig(BaseModel):
    module: str = Field(description="Module name for the vectorizer")
    class_name: str = Field(description="Class name for the vectorizer")
    model: str = Field(description="Path or name of the model")

    model_config = {"frozen": True}


class AppConfig(BaseModel):
    favorite_papers_path: str = Field(description="Path to the favorite papers JSON file")
    vectorizer: VectorizerConfig = Field(description="Vectorizer configuration")
    top_k: int = Field(default=10, description="Number of top recommendations to return")

    model_config = {"frozen": True}
