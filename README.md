# arXiv Recommender

`arxiv_recommender` is a content-based recommendation system that helps researchers discover relevant papers from arXiv. Given a set of favorite papers, it retrieves the most relevant ones using state-of-the-art NLP models.

## Project Structure
```
arxiv_recommender/
│── arxiv_paper_fetcher/      # Fetches arXiv paper metadata
│── text_vectorization/       # Handles text embedding models
│── recommendation/           # Core recommendation logic
│── utils/                    # Utility functions
│   │── json_handler.py       # JSON file handling
│   │── model_loader.py       # Dynamic text vectorizer loader
│   │── user_input.py         # Handles user-provided paper IDs
│── tests/                    # Unit tests for each module
│── data/                     # Stores favorite papers & config files
│   │── config.json           # User configuration file
│   │── favorite_papers.json  # Favorite paper metadata (auto-generated)
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies
│── setup.py                  # Package installation script
│── cli.py                    # Command-line interface
```

## Getting Started
### Installation
Ensure you have Python 3.8+ installed, then run:
```bash
git clone https://github.com/your-repo/arxiv_recommender.git
cd arxiv_recommender
pip install -r requirements.txt
```

### Configuration
Modify ` config.json` to set your preference:
```json
{
    "favorite_papers_path": "data/favorite_papers.json",
    "vectorizer": "text_vectorization.distill_bert.DistilBERTEmbedding",
    "top_k": 5
}
```
- `favorite_papers_path` -> Path to favorite papers file.
- `vectorizer` -> Embedding model.
- `top_k` -> Number of recommended papers.

### Running the CLI
You can run the arXiv Recommender CLI in one of the following ways:

#### Local Usage with Python

##### Environment Setup
We strongly recommend using a [Python virtual environment](https://docs.python.org/3/library/venv.html) to isolate dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -r requirements.txt
```
##### Running from Source (Module Mode)
Run the CLI directly from the source using Python’s module syntax:
```bash
python3 -m arxiv_recommender.bin.cli --config path/to/config.json
```
The --config flag is required.

If favorite_papers.json is missing or empty, the CLI will prompt you to enter arXiv paper IDs manually.

##### Running After Installation (via setup.py) (Working in Progress)
You’ll be able to install this project as a Python package and use the CLI globally:
```bash
pip install .
arxiv-recommender --config path/to/config.json
```
Setup instructions and entry point registration will be added in future versions.

#### Running via Docker (Working in Progress)
Docker provides a self-contained environment, so no Python or virtual environment is needed on your machine:
```bash
docker run --rm -v $(pwd)/data:/path/to/data arxiv-recommender --config /path/to/config.json
```
Docker support and prebuilt images are planned for a future release.

## License
This project is licensed under the MIT License.