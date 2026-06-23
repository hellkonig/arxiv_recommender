# ML Evaluation Specification

## Purpose

Evaluate the recommendation ranking shown to the user, not only the embedding
model in isolation. The evaluation should determine whether a proposed
embedding model or personal re-ranker improves the placement of papers the
user explicitly marks as interesting.

## Evaluation Unit

An evaluation example is a recommendation impression with:

- a paper identity
- the candidate set or recommendation run
- displayed rank and score
- embedding model and ranker versions
- selection source, including whether it was an exploration result
- timestamp
- optional explicit `interested` or `not_interested` feedback

Only explicitly labeled impressions are used as binary relevance judgments.
Unlabeled impressions are excluded rather than treated as negatives.

## Data Split

Use chronological evaluation:

```text
older labeled impressions -> fit or configure the candidate ranker
newer labeled impressions -> evaluate the frozen candidate ranker
```

Random splitting is not the default because it can leak future preferences and
does not reflect how the local recommender improves over time.

The split must prevent the same paper or feedback event from appearing in both
training and evaluation data.

## Metrics

### Primary Metric

`NDCG@10` is the initial primary ranking metric. It rewards placing explicitly
interested papers near the top of the list.

### Secondary Metrics

- `NDCG@5`
- `Precision@5`
- `Precision@10`
- pairwise ranking accuracy between labeled interested and not-interested
  papers from comparable recommendation contexts
- label coverage and number of evaluable recommendation runs
- ranking latency and embedding latency

Recall is not a primary metric because users may label only part of the
candidate set. It can be reported only for evaluation sets where all relevant
candidates have been judged.

Classification metrics such as ROC AUC may be reported for a personal
preference score, but they do not replace ranking metrics.

## Initial Comparisons

Evaluate these methods when sufficient feedback is available:

- TF-IDF baseline
- `BAAI/bge-small-en-v1.5`
- `allenai/specter2`
- the active base ranker with and without a personal re-ranker

Every comparison must use the same eligible candidate sets and labels. Record:

- model and code version
- text construction and preprocessing
- pooling and normalization
- ranking configuration
- feedback date range and counts
- hardware
- latency
- metric values

## Selection Bias

Historical feedback is biased because the active ranker controls which papers
the user sees. Offline comparison alone cannot completely correct this.

The product should eventually use controlled exploration and log the
probability or source of each explored selection. Until then, evaluation
results must be described as conditional on the collected impressions rather
than as an unbiased estimate over all arXiv papers.

## Promotion Policy

A candidate method must not replace the active method only because it has more
training data or greater complexity.

Before promotion, it must:

- improve the primary metric on later held-out feedback
- avoid a material regression in important secondary metrics
- have enough labeled examples and evaluable runs for a stable comparison
- meet local latency and resource constraints
- preserve a working fallback model

Exact minimum sample sizes and improvement thresholds will be set after the
application has collected enough data to estimate metric variability.

## Early-Stage Evaluation

Before enough user feedback exists:

- test deterministic ranking and embedding behavior
- verify normalization and cache-version correctness
- measure local runtime and memory use
- use TF-IDF and alternative models only as engineering baselines

Metadata proxies such as category overlap may be used for diagnostics, but
they are not treated as user-relevance ground truth.
