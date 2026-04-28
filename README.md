# arXiv Recommender

`arxiv_recommender` is a content-based recommendation system that helps researchers discover relevant papers from arXiv. Given a set of favorite papers, it retrieves the most relevant ones using state-of-the-art NLP models.

## Project Structure

```
arxiv_recommender/
├── src/
│   └── arxiv_recommender/           # Python package
│       ├── cli.py                    # CLI entry point
│       ├── arxiv_paper_fetcher/      # Fetches arXiv paper metadata
│       ├── text_vectorization/       # Handles text embedding models
│       ├── recommendation/           # Core recommendation logic
│       ├── schemas/                  # Pydantic models
│       └── utils/                    # Utility functions
├── configs/
│   └── config.json.example           # Configuration template
├── tests/                            # Unit tests
├── pyproject.toml                    # Project configuration
└── README.md                         # Project documentation
```

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/your-repo/arxiv_recommender.git
cd arxiv_recommender
pip install -e ".[dev]"
```

### 2. Create Configuration

Copy the example config and customize:

```bash
cp configs/config.json.example configs/config.json
```

## Configuration

Edit `configs/config.json`:

```json
{
    "favorite_papers_path": "favorite_papers.json",
    "vectorizer": {
        "module": "distil_bert",
        "class_name": "DistilBERTEmbedding",
        "model": "distilbert-base-uncased"
    },
    "top_k": 10
}
```

| Field | Description |
|-------|-------------|
| `favorite_papers_path` | Path to favorite papers JSON file |
| `vectorizer.module` | Module name for vectorizer |
| `vectorizer.class_name` | Class name for vectorizer |
| `vectorizer.model` | Model name or local path |
| `top_k` | Number of recommended papers |

## Custom Models

### Using a Custom Model

1. Place your model in a local directory (e.g., `./models/my-model/`)
2. Update `configs/config.json`:

```json
{
    "vectorizer": {
        "model": "./models/my-model"
    }
}
```

### Model Cache Location

Models are cached at `~/.cache/huggingface/hub/`. To use a custom location:

```bash
export HF_HOME=/your/custom/path
```

## Running the CLI

### Option 1: Using Entry Point (after installation)

```bash
arxiv-recommend --config configs/config.json
```

### Option 2: Running from Source

```bash
pip install -e .
python -m arxiv_recommender.cli --config configs/config.json
```

### Options

| Flag | Description |
|------|-------------|
| `--config` | Path to configuration JSON file (required) |
| `--date_of_pulling_papers` | Date in YYYYMMDD format (optional, defaults to today) |

If `favorite_papers.json` is missing or empty, the CLI will prompt you to enter arXiv paper IDs.

## Testing

```bash
pip install -e ".[test]"
python -m pytest
```

## License

This project is licensed under the MIT License.