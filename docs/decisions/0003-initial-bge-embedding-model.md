# ADR 0003: Use BGE-Small as the Initial Embedding Model

- Status: Accepted
- Date: 2026-06-23

## Context

The current implementation uses raw `distilbert-base-uncased` outputs with
mean pooling. DistilBERT and raw SciBERT are language-model backbones rather
than models specifically trained to produce text representations for semantic
retrieval.

The application needs an embedding model that can run locally, represent paper
titles and abstracts, and support efficient cosine-similarity ranking.

## Decision

Use `BAAI/bge-small-en-v1.5` as the initial frozen embedding model.

The implementation will:

- embed the paper title and abstract together
- follow the model's recommended pooling behavior
- L2-normalize output embeddings
- compare normalized embeddings with cosine similarity or an equivalent dot
  product
- record the model identifier and embedding configuration with persisted
  results

This decision requires an implementation change rather than only replacing
the configured model name, because pooling and normalization are part of the
embedding contract.

## Alternatives Considered

- Raw `distilbert-base-uncased`
- Raw `allenai/scibert_scivocab_uncased`
- `allenai/specter2`
- TF-IDF

## Consequences

- The initial model remains small enough for ordinary local hardware.
- Existing cached embeddings must not be reused across model or pooling
  versions.
- SPECTER2 remains a scientific-domain candidate for later comparison.
- TF-IDF remains a low-cost non-neural baseline.
- Model quality is provisional until explicit user feedback supports a
  chronological comparison.

## Revisit When

Revisit the active embedding model after enough explicit feedback exists to
compare BGE-small, SPECTER2, and TF-IDF on the same historical recommendation
events.
