# AGENTS.md - Project Configuration

## Project Overview
Content-based arXiv paper recommender system using DistilBERT embeddings.

## Commands
```bash
uv run python -m pytest            # Run tests (uv)
uv run ruff check .                 # Run linting (uv)
uv run python -m mypy               # Run type checking (uv)
uv run python -m pytest tests/path/to/test_file.py::TestClass::test_method  # Single test
python3 -m arxiv_recommender.cli --config path/to/config.json   # Run CLI
```

## Project Structure
```
arxiv_recommender/
├── arxiv_recommender/
│   ├── recommendation/         # Core recommendation logic
│   ├── text_vectorization/    # Embedding models (DistilBERT)
│   ├── arxiv_paper_fetcher/  # arXiv API client
│   ├── schemas/              # Pydantic models (NEW)
│   └── utils/                # Utilities
├── tests/                    # Test suite
└── cli.py                   # CLI entry point
```

## Conventions
- Follows PEP 8 and Google Python Style Guide (compatible, not conflicting)
- **Type hints (Python 3.9+)**: Use built-in generics (`list[str]`, `dict[str, int]`)
- **Type hints (Python 3.10+)**: Use `X | None` instead of `Optional[X]`
- Classes: PascalCase, Functions: snake_case
- Google-style docstrings for public methods

## Three-Tier Boundaries
| Always | Ask First | Never |
|--------|----------|-------|
| Run tests before commit | Add new dependencies | Edit `requirements.txt` directly |
| Validate with Pydantic | Modify `schemas/` models | Push to main branch |
| Use type hints | - | Skip tests when adding features |

## Git Workflow
- Branch: `feat/<name>`, `fix/<name>`, `chore/<name>`
- Commit: `<type>: <description>` (e.g., `feat: add user auth`)
- PR title: `feat: add user auth`
- **Before every push**: Run local CI (lint + typecheck + tests):
  ```bash
  uv run ruff check . && uv run python -m mypy && uv run python -m pytest
  ```

## Prohibitions
- **Never** commit secrets: `.env`, `credentials.json`
- **Never** edit `requirements.txt` - use `pyproject.toml`
- **Never** edit `__pycache__` - auto-generated

**IMPORTANT**: Run tests before committing. Validate inputs with Pydantic.