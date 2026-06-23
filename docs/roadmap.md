# Project Roadmap

This roadmap tracks the implementation sequence for the local personalized
recommender. Accepted decisions are recorded separately in `docs/decisions/`.

## Milestone 1: Retrieval Embedding Foundation

- [ ] Integrate `BAAI/bge-small-en-v1.5` with model-appropriate pooling.
- [ ] L2-normalize embeddings and verify similarity behavior.
- [ ] Version cache entries by model and embedding configuration.
- [ ] Add focused unit tests for pooling, normalization, and caching.
- [ ] Measure embedding latency on representative title-and-abstract inputs.

## Milestone 2: Local Feedback Vertical Slice

- [ ] Define and review the SQLite schema and migration approach.
- [ ] Store papers, recommendation runs, and impressions.
- [ ] Store explicit `interested` and `not_interested` feedback.
- [ ] Record embedding model, ranker version, displayed rank, and score.
- [ ] Build a thin local web interface that shows recommendation details.
- [ ] Add interested and not-interested controls with clear saved/error states.
- [ ] Add persistence and workflow tests.

## Milestone 3: Historical Evaluation

- [ ] Implement chronological dataset construction from local feedback.
- [ ] Implement NDCG@5, NDCG@10, Precision@5, and Precision@10.
- [ ] Implement pairwise ranking accuracy where labels are comparable.
- [ ] Report label coverage, sample counts, and runtime with every result.
- [ ] Add a reproducible local evaluation command.

## Milestone 4: Personal Re-Ranking

- [ ] Establish the base-ranking evaluation result.
- [ ] Implement a simple personal re-ranker on frozen embeddings.
- [ ] Train candidate rankers without blocking the recommendation workflow.
- [ ] Keep the current ranker as a fallback.
- [ ] Promote a candidate only after chronological evaluation.

## Milestone 5: Model Comparison and Exploration

- [ ] Add a TF-IDF retrieval baseline.
- [ ] Add an `allenai/specter2` experiment path.
- [ ] Compare BGE-small, SPECTER2, and TF-IDF on identical eligible data.
- [ ] Define and implement a controlled exploration policy.
- [ ] Record selection source and exploration metadata.

## Later Considerations

- [ ] Evaluate a local FAISS index if the historical corpus outgrows exact
  NumPy similarity.
- [ ] Evaluate adapters or LoRA only if feedback volume and evaluation results
  justify transformer adaptation.
- [ ] Define local data export, deletion, backup, and recovery workflows.
- [ ] Reassess hosted or synchronized architecture only if product scope
  changes.
