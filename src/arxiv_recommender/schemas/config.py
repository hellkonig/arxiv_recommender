from pydantic import BaseModel, Field


class VectorizerConfig(BaseModel):
    module_name: str = Field(description="Module name for the vectorizer")
    class_name: str = Field(description="Class name for the vectorizer")
    model_name: str = Field(description="Path or name of the model")
    cache_size: int = Field(default=1000, description="Maximum number of embeddings to cache")

    model_config = {"frozen": True}


class AppConfig(BaseModel):
    favorite_papers_path: str = Field(description="Path to the favorite papers JSON file")
    vectorizer: VectorizerConfig = Field(description="Vectorizer configuration")
    top_k: int = Field(default=10, description="Number of top recommendations to return")

    model_config = {"frozen": True}
