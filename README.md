# arXiv Recommender

`arxiv_recommender` is a content-based recommendation system that helps researchers discover relevant papers from arXiv. Given a set of favorite papers, it fetches candidate papers from arXiv and ranks them using configurable HuggingFace text embeddings.

## Project Structure

```
arxiv_recommender/
├── src/
│   └── arxiv_recommender/           # Python package
│       ├── interfaces/               # CLI and other user-facing interfaces
│       ├── arxiv_paper_fetcher/      # Fetches arXiv paper metadata
│       ├── favorite_papers/           # Loads favorite papers from files or prompts
│       ├── text_vectorization/       # Handles text embedding models
│       ├── recommendation/           # Core recommendation logic
│       ├── persistence/              # Local SQLite schema and migrations
│       ├── schemas/                  # Pydantic models
│       └── utils/                    # Utility functions
├── configs/
│   └── config.json.example           # Configuration template
├── benchmarks/                       # Local engineering benchmarks
├── tests/                            # Unit tests
├── pyproject.toml                    # Project configuration
└── README.md                         # Project documentation
```

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/your-repo/arxiv_recommender.git
cd arxiv_recommender
uv sync --extra dev
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
        "module_name": "huggingface_embed",
        "class_name": "HuggingFaceEmbedding",
        "model_name": "BAAI/bge-small-en-v1.5",
        "cache_size": 1000,
        "pooling_strategy": "auto",
        "normalize_embeddings": "auto",
        "max_length": 512
    },
    "top_k": 10,
    "log_level": "INFO"
}
```

| Field | Description |
|-------|-------------|
| `favorite_papers_path` | Path to favorite papers JSON file |
| `vectorizer.module_name` | Module name for vectorizer |
| `vectorizer.class_name` | Class name for vectorizer |
| `vectorizer.model_name` | Model name or local path |
| `vectorizer.cache_size` | Maximum number of embeddings to cache |
| `vectorizer.pooling_strategy` | Embedding pooling strategy (`auto`, `mean`, or `cls`) |
| `vectorizer.normalize_embeddings` | Whether to L2-normalize embeddings (`auto`, `true`, or `false`) |
| `vectorizer.max_length` | Maximum token length for embedding inputs |
| `top_k` | Number of recommended papers |
| `log_level` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`) |

## Custom Models

### Using a Custom Model

1. Place your model in a local directory (e.g., `./models/my-model/`)
2. Update `configs/config.json`:

```json
{
    "vectorizer": {
        "model_name": "./models/my-model"
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
uv run python -m arxiv_recommender.interfaces.cli --config configs/config.json
```

### Options

| Flag | Description |
|------|-------------|
| `--config` | Path to configuration JSON file (required) |
| `--date_of_pulling_papers` | Date in YYYYMMDD format (optional, defaults to today) |
| `--log-level` | Override configured log level |
| `--stats` | Print a metrics summary at the end of execution |

If `favorite_papers.json` is missing or empty, the CLI will prompt you to enter arXiv paper IDs.

## Testing

```bash
uv run python -m pytest
```

## Benchmarking

Measure embedding latency for representative title-and-abstract inputs:

```bash
uv run python benchmarks/embedding_latency.py \
  --config configs/config.json \
  --sizes 1 10 100 \
  --repeats 3
```

Benchmark results are local-machine-specific and are intended for comparing
future model or embedding-pipeline changes.

## License

This project is licensed under the MIT License.
