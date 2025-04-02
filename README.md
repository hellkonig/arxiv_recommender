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

### COnfiguration
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
To start the recommendation process, run:
```bash
python cli.py --config data/config.json
```
If favorite_papers.json does not exist, the CLI will prompt you to enter arXiv paper IDs.

## License
This project is licensed under the MIT License.