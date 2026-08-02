# Personalized Recommender Design

## Status

This is a living design document. The accepted architectural decisions are
recorded in `docs/decisions/`.

## Product Objective

Build a local application that retrieves recent arXiv papers, ranks them for
an individual user, collects explicit preference feedback, and improves future
rankings without sending preference data to a centralized service.

## Product Principles

- Local-first: recommendation history and feedback remain on the user's
  computer.
- Simple interaction: a shown paper can be marked `interested` or
  `not_interested`.
- Automatic personalization: the application chooses and updates the ranking
  method without requiring ML knowledge from the user.
- Measured changes: a candidate ranker replaces the active ranker only after
  chronological evaluation indicates an improvement.
- Transparent provenance: recommendation events record the model, ranker, and
  configuration that produced them.

## Initial Scope

The initial user workflow is:

1. Load the user's existing favorite papers.
2. Fetch candidate papers from arXiv.
3. Embed and rank the candidates.
4. Display recommendations in a local web interface.
5. Let the user mark each shown paper as `interested` or `not_interested`.
6. Store impressions and explicit feedback in a local SQLite database.
7. Use accumulated feedback to evaluate and later personalize rankings.

The initial scope does not include accounts, centralized storage, cross-device
sync, social recommendations, or full transformer fine-tuning.

## Recommendation Pipeline

The initial ranking pipeline is:

```text
favorite papers + candidate papers
              |
              v
frozen BGE-small embedding model
              |
              v
cosine similarity to favorite-paper embeddings
              |
              v
base ranking
              |
              v
optional personal re-ranker
              |
              v
final ranking and controlled exploration
              |
              v
local web interface
```

`BAAI/bge-small-en-v1.5` is the initial embedding model. Paper input consists
of the title and abstract. Embeddings are L2-normalized and cached with an
identifier that includes the model and embedding configuration.

Embedding configuration is resolved before inference. User-facing config may
use `auto` for pooling and normalization, but runtime embedding metadata uses
the resolved values. The BGE-small profile resolves to CLS pooling with
L2-normalized embeddings. Known model profiles reject incompatible explicit
settings so cached embeddings and later evaluation remain reproducible.

## Feedback Semantics

Only explicit binary feedback is used as a training or evaluation label:

- `interested`: the user considers the paper relevant or useful enough to
  pursue.
- `not_interested`: the user explicitly rejects the paper.

No interaction is an unlabeled event. It must not be converted automatically
into `not_interested`.

Feedback may change. The persistence layer should retain timestamps and define
whether the latest explicit response or an event history is used by a given
training job.

## Local Persistence

SQLite will store structured application state. The initial conceptual tables
are:

- `papers`: arXiv identity and paper metadata
- `recommendation_runs`: request context and candidate-set information
- `impressions`: paper, displayed rank, scores, and selection source
- `feedback`: explicit interested/not-interested events
- `model_versions`: embedding and ranker identifiers and configuration

The initial schema, migration runner, and recommendation repository are
implemented under `src/arxiv_recommender/persistence/`. Repository inputs and
outputs are validated with Pydantic, and each recommendation run is stored in
one transaction so partial model, paper, run, or impression records are rolled
back together. Only papers selected for display are retained as papers and
impressions; the total fetched candidate count remains part of the run record.
Embedding and ranker version records are reused through canonical configuration
JSON. The repository is not connected to the CLI workflow yet.

Persisted embeddings must be invalidated or separated when the embedding model,
pooling, normalization, or text-construction policy changes.

## Personalization Strategy

Personalization is introduced incrementally:

1. Use only the base similarity ranking while feedback is sparse.
2. Train a lightweight personal re-ranker on frozen paper embeddings.
3. Evaluate candidate updates using older feedback for fitting and later
   feedback for evaluation.
4. Promote a candidate only when it improves the agreed ranking metrics and
   does not introduce unacceptable runtime or stability regressions.
5. Consider adapter or LoRA training only after substantially more feedback
   exists and a lightweight ranker no longer provides sufficient improvement.

The first personal ranker should be deliberately simple, such as a centroid
preference score or regularized logistic regression. Exact thresholds for
training or promotion must be evidence-based rather than presented as user
settings.

## Exploration

Feedback collected only from the active model's top results is biased toward
that model. The final recommendation list should eventually reserve a small,
configurable share of impressions for controlled exploration.

Exploration candidates may come from:

- lower positions in the active ranking
- an alternative retrieval model
- a diversity-aware selection step

Exploration must be logged, and its exact policy will be defined after the
initial feedback workflow is usable.

## Privacy and Operations

- Feedback and recommendation history remain local by default.
- The application must not log full private data unnecessarily.
- The database location and deletion/export behavior must be documented.
- Downloaded model files follow the configured model provider's cache policy.
- Schema migrations must preserve user feedback or fail with a recoverable
  backup path.

## Related Decisions

- `docs/decisions/0001-local-first-architecture.md`
- `docs/decisions/0002-explicit-feedback-and-personalization.md`
- `docs/decisions/0003-initial-bge-embedding-model.md`
- `docs/decisions/0004-sqlite-local-storage.md`
