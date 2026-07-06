# Code Review Snapshot

Objective review of the current `arxiv_recommender` codebase, focused on correctness, maintainability, ML quality, testing, and operational readiness.

- Reviewed on: 2026-05-25
- Updated on: 2026-07-07 after PR #35
- Scope: current codebase snapshot
- Context: local CLI or localhost app priority

This document is a review snapshot, not a final verdict. Priority should follow the immediate product goal. If the next target is a usable local CLI application or a localhost web app, end-to-end reliability and retrieval quality should come before scalability work.

## Current Verification Status

- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- `uv run python -m mypy src`: passed.
- `uv run python -m pytest`: passed, 138 tests.

## Priority Findings

These findings describe the main engineering and ML risks in the current codebase. Their importance does not change, but the order of implementation should follow the product goal. For a local user-facing app, retrieval quality still matters; batching and larger-scale performance tuning can move later if current runtime is acceptable.

### Partially Resolved: Recommendation Quality Is Not ML-Validated

PR #35 replaced the default raw DistilBERT configuration with a BGE-small
embedding profile. `BAAI/bge-small-en-v1.5` now resolves to CLS pooling with
L2-normalized embeddings, and cache entries are versioned by resolved embedding
configuration.

The remaining gap is evaluation: recommendation quality is still not validated
against explicit user feedback or a retrieval benchmark.

Recommended improvements:

- Add an offline evaluation harness with labeled or proxy relevance data.
- Track metrics such as recall@k, NDCG@k, MRR, and diversity.
- Compare against simple baselines such as TF-IDF/BM25 before changing models.
- Compare BGE-small with SPECTER2 and TF-IDF after enough eligible feedback
  exists.

Relevant code:

- `src/arxiv_recommender/text_vectorization/huggingface_embed.py`
- `src/arxiv_recommender/recommendation/recommendation.py`

### High: Embedding Inference Does Not Scale

The recommender embeds favorite papers and candidate papers one at a time. For daily arXiv pulls with up to thousands of papers, this causes many small tokenizer/model calls and poor throughput.

Recommended improvements:

- Add a batch embedding interface to `TextEmbedder`.
- Tokenize and infer candidate papers in batches.
- Make batch size configurable.
- Measure latency before and after the change.

Relevant code:

- `src/arxiv_recommender/recommendation/recommendation.py`
- `src/arxiv_recommender/text_vectorization/base.py`
- `src/arxiv_recommender/text_vectorization/huggingface_embed.py`

### Resolved: Fetcher Tests Contained False Positives

`ArxivFetcher` calls `response.raise_for_status()`, so real HTTP 404 responses should raise. Some tests expect `None` because the mock response does not configure `raise_for_status`, which means the test does not reflect production behavior.

Status:

- Resolved by PR #29, `fix: clarify arxiv fetcher error semantics`.
- Mocked responses now configure `raise_for_status` explicitly.
- HTTP failures raise `requests` exceptions after retry behavior is applied.
- Valid empty feeds remain distinct from failures: `get_paper_by_id()` returns `None`, and `get_daily_papers()` returns `[]`.
- Malformed arXiv XML now raises `ArxivParseError` instead of being treated as an empty result.

Relevant code:

- `src/arxiv_recommender/arxiv_paper_fetcher/fetcher.py`
- `src/arxiv_recommender/arxiv_paper_fetcher/parser.py`
- `tests/arxiv_paper_fetcher/test_fetcher.py`
- `tests/arxiv_paper_fetcher/test_parser.py`

### Medium: Metrics Are Partially Unwired

`MetricsCollector` tracks API calls, API errors, retry attempts, cache hits, and cache misses, but most of these counters are not updated by the fetcher, retry decorator, or embedding cache.

Recommended improvements:

- Wire API call/error counters into `ArxivFetcher`.
- Wire retry counters into `retry_with_backoff`.
- Either remove duplicate cache counters from `MetricsCollector` or update them from `EmbeddingCache`.
- Add tests that verify metrics reflect real operations.

Relevant code:

- `src/arxiv_recommender/utils/metrics.py`
- `src/arxiv_recommender/utils/retry.py`
- `src/arxiv_recommender/arxiv_paper_fetcher/fetcher.py`
- `src/arxiv_recommender/text_vectorization/cache.py`

### Resolved: Recommendation Outputs Lacked Paper Identity

`Paper` previously only included title and abstract, and recommendations returned loose dictionaries. This was resolved by PR #31, `feat: add structured paper metadata and typed recommendations`.

Status:

- `Paper` now includes optional arXiv ID, URL, authors, categories, published timestamp, and updated timestamp.
- arXiv Atom parsing extracts paper identity and metadata.
- Recommendation results now use typed `RecommendationItem` models instead of `dict[str, Any]`.
- Metadata is preserved through parsing, ranking, pipeline output, and CLI formatting.
- Favorite-paper JSON loading validates untrusted JSON shape before Pydantic model validation.
- Python 3.10-compatible UTC timestamp parsing is covered by tests.

Relevant code:

- `src/arxiv_recommender/schemas/paper.py`
- `src/arxiv_recommender/arxiv_paper_fetcher/parser.py`
- `src/arxiv_recommender/recommendation/types.py`
- `src/arxiv_recommender/recommendation/recommendation.py`

### Medium: Config Validation Is Too Permissive

Pydantic models are used, and PR #30 tightened deterministic scalar constraints while adding runtime validation for environment-dependent vectorizer settings.

Status:

- Partially addressed by PR #30, `fix: validate configuration and vectorizer loading`.
- `top_k > 0` is enforced.
- `cache_size >= 0` is enforced, allowing zero to disable caching.
- `log_level` is normalized and restricted to known logging levels.
- CLI log-level choices and logging setup reuse the same supported-level constant.
- Invalid vectorizer modules, classes, dependencies, and model-instantiation failures now raise actionable `VectorizationModelLoadError` messages.

Design decision:

- `module_name`, `class_name`, and `model_name` are validated when the vectorizer is loaded instead of using schema-level non-empty-string checks.
- Non-empty-string checks would reject only one trivial failure mode while arbitrary invalid names would still fail later.
- Runtime validation can distinguish missing modules, missing classes, missing dependencies, and model-instantiation failures and return actionable errors.

Relevant code:

- `src/arxiv_recommender/schemas/config.py`
- `src/arxiv_recommender/utils/logging.py`
- `src/arxiv_recommender/utils/model_loader.py`
- `tests/schemas/test_config.py`
- `tests/utils/test_logging.py`
- `tests/utils/test_model_loader.py`

### Resolved: arXiv Query Construction Was Brittle

Daily-paper query construction previously assembled arXiv API URLs manually, and `max_results` was included in both the formatted query and fetcher URL. This was resolved by PR #32, `fix: validate arxiv query parameters`.

Status:

- `ArxivFetcher` now calls `requests.get(..., params=...)` for paper ID and daily-paper requests.
- Daily-paper query params are built as structured data instead of URL fragments.
- `max_results` is passed once as a request parameter.
- arXiv date validation enforces exact `YYYYMMDD` shape and real calendar dates.
- arXiv category validation rejects malformed category/query-fragment inputs.
- CLI date parsing validates before pipeline setup and preserves clear argparse errors.
- Config loading moved out of `cli.py` into `utils.config_loader`, keeping CLI orchestration thinner.

Relevant code:

- `src/arxiv_recommender/arxiv_paper_fetcher/utils.py`
- `src/arxiv_recommender/arxiv_paper_fetcher/fetcher.py`
- `src/arxiv_recommender/cli.py`
- `src/arxiv_recommender/utils/config_loader.py`
- `tests/arxiv_paper_fetcher/test_utils.py`
- `tests/arxiv_paper_fetcher/test_fetcher.py`
- `tests/test_cli.py`
- `tests/utils/test_config_loader.py`

### Low: Dependency And Packaging Workflow Is Inconsistent

The repository has `pyproject.toml`, `uv.lock`, and `requirements.txt`. `.gitignore` ignores `uv.lock`, but `uv.lock` is tracked. Project instructions also prohibit editing `requirements.txt` directly.

Recommended improvements:

- Choose one primary dependency workflow.
- If using `uv`, keep `uv.lock` tracked and remove it from `.gitignore`.
- Treat `requirements.txt` as generated or remove it if not needed.
- Document the canonical install/test workflow in `README.md`.

Relevant files:

- `pyproject.toml`
- `uv.lock`
- `requirements.txt`
- `.gitignore`
- `README.md`

## Suggested Improvement Order

The following order is recommended if the immediate goal is a usable local CLI app or a web app served on `localhost`.

1. Add a lightweight retrieval quality evaluation harness.
2. Improve CLI error reporting for fetcher and parser failures.
3. Improve the CLI workflow or build a thin localhost web UI.
4. Wire metrics fully or simplify them to only report trustworthy values.
5. Add batch embedding support and benchmark latency.
6. Clean up dependency and packaging workflow.

## Local App Priority Notes

### Retrieval Quality Still Matters

Even for a local app, recommendation quality is still part of the core product. A fast local tool that returns weak recommendations is not useful enough to justify the workflow.

Recommended improvements:

- Build a small benchmark set using representative favorite-paper inputs.
- Compare the current embedding approach against simple baselines such as TF-IDF or BM25.
- Track retrieval metrics such as recall@k, NDCG@k, MRR, and diversity.
- Use these results to justify model changes instead of changing models by intuition alone.

### Scalability Can Move Later

For a first local app version, scalability work can be deferred if the runtime is acceptable on realistic local usage. The main risk is not that the system cannot scale yet; it is that users cannot tell whether the recommendations are good.

Recommended improvements:

- Keep batch embedding and throughput tuning on the roadmap.
- Measure runtime on a realistic daily candidate set before optimizing further.
- Prioritize batching once the app workflow and retrieval quality are understood.

### Immediate Local App Outcomes

If building toward a local user-facing app, the next concrete bar should be:

- clear API and input errors
- validated configuration, CLI date, and arXiv query inputs, now available after PR #32
- recommendation outputs with paper identity and links, now available after PR #31
- a lightweight but real retrieval evaluation loop
- a usable CLI or localhost interface
- trustworthy logs and metrics

## Engineering Bar

The current codebase is clean enough for a small prototype: linting, formatting, type checking, and tests pass. The main gap is not style; it is product and ML reliability.

For the immediate local app goal, the next bar is a reliable end-to-end workflow with clear errors, validated inputs, traceable recommendation outputs, and a lightweight retrieval evaluation loop. After that, the project should improve scalability and stronger ML benchmarking before recommendations are treated as trustworthy at larger scale.
