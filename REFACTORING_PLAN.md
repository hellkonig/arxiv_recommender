# Refactoring Plan - arxiv_recommender

**Created**: 2026-03-14  
**Status**: Phase 1 COMPLETED

---

## Vision

Transform this codebase from a prototype to production-ready ML system following top-tier ML engineering standards.

---

## Branching Strategy

```
main (protected)
  └── feature/phase1-infrastructure → PR → main ✅ DONE
  └── feature/phase2-validation → PR → main
  └── feature/phase3-reliability → PR → main
  └── feature/phase4-caching → PR → main
  └── feature/phase5-observability → PR → main
  └── feature/phase6-pipeline → PR → main
```

Each phase = 1 branch → 1 PR → merge to main

---

## Current Issues

| File | Issue | Severity | Status |
|------|-------|----------|--------|
| `recommendation.py` | No caching, sequential embedding computation | High | - |
| `fetcher.py` | No retry logic, hardcoded timeout | High | - |
| All modules | No Pydantic validation on inputs | High | Phase 2 |
| `distil_bert.py` | No batch processing, loads model eagerly | Medium | - |
| `parser.py` | Silent exception swallowing | Medium | ✅ Fixed |
| `cli.py` | No structured logging | Medium | In progress |
| All modules | No type-safe return types | Low | ✅ Fixed |
| Project | No pyproject.toml | High | ✅ Fixed |

---

# Phase Instructions

Each phase includes step-by-step instructions. Start new session from `main` branch.

---

## Phase 1: Infrastructure (pyproject.toml + CI) ✅ DONE

**Goal**: Set up modern Python project structure with automated testing

### Completed Steps

- Created branch `feature/phase1-infrastructure`
- Created pyproject.toml (PEP 621 compliant)
- Created CI workflow with ruff + mypy + pytest
- Added .pre-commit-config.yaml
- Applied ruff formatting
- Fixed mypy type errors
- Addressed PR comments

### Files Created/Modified

| File | Status |
|------|--------|
| `pyproject.toml` | Created - PEP 621 with ruff, mypy, pytest |
| `.github/workflows/ci.yml` | Created - lint + test jobs using uv |
| `.pre-commit-config.yaml` | Created - ruff + mypy hooks |
| `uv.lock` | Added - for reproducible CI |
| `.gitignore` | Updated - .venv, .ruff_cache, etc. |
| Type annotations | Fixed - modern `list[str]` syntax |
| `recommendation.py` | Fixed - top_k bug |

### Next Phase

## Phase 2: Data Validation & Config

**Goal**: Add type safety and input validation

### Steps

```bash
# 1. Create branch
git checkout -b feature/phase2-validation main

# 2. Create schemas/
mkdir -p arxiv_recommender/schemas

# 3. Files to modify/create
# - arxiv_recommender/schemas/__init__.py
# - arxiv_recommender/schemas/paper.py     # Paper model
# - arxiv_recommender/schemas/config.py   # AppConfig model
# - cli.py                             # Use AppConfig
# - parser.py                          # Return Paper objects

# 4. Run tests
python -m pytest

# 5. Commit & push
git add .
git commit -m "feat: add Pydantic schemas for validation"
git push -u origin feature/phase2-validation
```

### Files

| File | Description |
|------|-------------|
| `schemas/__init__.py` | Exports: Paper, AppConfig |
| `schemas/paper.py` | Pydantic model for paper metadata |
| `schemas/config.py` | Pydantic model for app config |

---

## Phase 3: Resilience & Reliability

**Goal**: Handle external API failures gracefully

### Steps

```bash
# 1. Create branch
git checkout -b feature/phase3-reliability main

# 2. Create utils/retry.py
# - Add @retry_with_backoff decorator

# 3. Modify fetcher.py
# - Apply retry decorator
# - Configurable timeout

# 4. Run tests & commit
```

### Files

| File | Description |
|------|-------------|
| `utils/retry.py` | Retry decorator with exponential backoff |
| `fetcher.py` | Apply retry logic |

---

## Phase 4: Performance & Caching

**Goal**: Avoid recomputing embeddings

### Files

| File | Description |
|------|-------------|
| `text_vectorization/cache.py` | EmbeddingCache class |
| `recommendation.py` | Use cache |

---

## Phase 5: Observability

**Goal**: Production-grade logging and metrics

### Files

| File | Description |
|------|-------------|
| `utils/logging.py` | JSON formatter |
| `utils/metrics.py` | Metrics collection |

---

## Phase 6: Pipeline Abstraction

**Goal**: Enable testing and swapping components

### Files

| File | Description |
|------|-------------|
| `recommendation/base.py` | Abstract base class |
| `recommendation/pipeline.py` | Pipeline implementation |

---

## Quick Reference

### Commands

```bash
# Start new phase
git checkout -b feature/phase<N>-<name> main

# Install dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Run single test
python -m pytest tests/path/to/test.py::TestClass::test_method
```

### Important Files

- `AGENTS.md` - Always read before working
- `README.md` - Project documentation
- This file - Reference for current phase

---

## When Complete

Delete this file after all phases are done:
```bash
rm REFACTORING_PLAN.md
```