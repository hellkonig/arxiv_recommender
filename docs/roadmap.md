# Project Roadmap

This roadmap tracks the implementation sequence for the local personalized
recommender. Accepted decisions are recorded separately in `docs/decisions/`.

## Milestone 1: Retrieval Embedding Foundation

- [x] Integrate `BAAI/bge-small-en-v1.5` with model-appropriate pooling.
- [x] L2-normalize embeddings and verify similarity behavior.
- [x] Version cache entries by model and embedding configuration.
- [x] Add focused unit tests for pooling, normalization, and caching.
- [x] Measure embedding latency on representative title-and-abstract inputs.

## Milestone 2: Local Feedback Vertical Slice

- [x] Define and review the SQLite schema and migration approach.
- [ ] Store papers, recommendation runs, and impressions.
- [ ] Store explicit `interested` and `not_interested` feedback.
- [ ] Record embedding model, ranker version, displayed rank, and score.
- [ ] Build a thin local web interface that shows recommendation details.
- [ ] Add interested and not-interested controls with clear saved/error states.
- [ ] Add persistence and workflow tests.

### Planned Implementation PRs

#### PR 1: Recommendation Persistence Repository

Suggested branch: `feat/recommendation-persistence`

- [x] Add validated persistence input and output models.
- [x] Add a repository for recommendation runs.
- [x] Store only papers included in the displayed recommendation list.
- [x] Insert or reuse embedding and ranker model versions.
- [x] Insert recommendation runs and displayed impressions atomically.
- [x] Return structured recommendation-run and impression identifiers.
- [x] Roll back the complete operation when any persistence step fails.
- [x] Add integration tests using temporary migrated SQLite databases.

Status: Implemented in PR #39.

The repository API is complete, but the top-level persistence item remains open
until PR 3 wires it into the CLI workflow.

#### PR 2: Ranking Provenance

Suggested branch: `feat/ranking-provenance`

- [x] Define developer-maintained metadata for the base ranker.
- [x] Record its algorithm name, semantic version, and canonical configuration.
- [x] Expose resolved embedding provenance through the text-embedding interface.
- [x] Store resolved pooling, normalization, maximum length, and text policy.
- [x] Add a typed impression selection source.
- [x] Update embedding implementations, test doubles, and provenance tests.

Status: Implemented in PR #40.

Ranker versions are maintained by developers rather than configured by users.
They change when scoring semantics or score-affecting implementation behavior
changes.

#### PR 2A: Pin Hugging Face Model Revisions

Suggested branch: `feat/pin-embedding-revisions`

- Add validated Hugging Face model-revision configuration.
- Load the tokenizer and model from the same immutable artifact revision.
- Record artifact revision separately from embedding implementation behavior.
- Include both identities in embedding cache namespaces.
- Add configuration, loading, provenance, and documentation tests.

This follow-up must land before PR 3 persists CLI recommendation runs so
historical model-version records cannot conflate different upstream weights.

#### PR 3: Persist CLI Recommendation Runs

Suggested branch: `feat/persist-cli-runs`

- Add a configurable local database path.
- Capture run start time, completion time, and effective requested date.
- Connect to the database and apply migrations during CLI startup.
- Persist a completed run before displaying its recommendations.
- Store displayed rank, score, selection source, metrics, and model-version
  references.
- Add CLI and end-to-end workflow tests.
- Document the database location and displayed-paper retention policy.

After this PR, CLI recommendation history is durable. Explicit feedback
controls remain a separate subsequent roadmap item.

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
