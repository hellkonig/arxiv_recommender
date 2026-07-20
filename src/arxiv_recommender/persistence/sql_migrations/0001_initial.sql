CREATE TABLE papers (
    id INTEGER PRIMARY KEY,
    arxiv_id TEXT UNIQUE,
    url TEXT,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    published_at TEXT,
    updated_at TEXT,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY,
    model_kind TEXT NOT NULL CHECK(model_kind IN ('embedding', 'ranker')),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(model_kind, name, version, config_json)
);

CREATE TABLE recommendation_runs (
    id INTEGER PRIMARY KEY,
    run_started_at TEXT NOT NULL,
    run_completed_at TEXT,
    requested_date TEXT,
    favorite_papers_count INTEGER NOT NULL CHECK(favorite_papers_count >= 0),
    candidate_papers_count INTEGER NOT NULL CHECK(candidate_papers_count >= 0),
    top_k INTEGER NOT NULL CHECK(top_k > 0),
    embedding_model_version_id INTEGER NOT NULL,
    ranker_model_version_id INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    FOREIGN KEY(embedding_model_version_id) REFERENCES model_versions(id),
    FOREIGN KEY(ranker_model_version_id) REFERENCES model_versions(id)
);

CREATE TABLE impressions (
    id INTEGER PRIMARY KEY,
    recommendation_run_id INTEGER NOT NULL,
    paper_id INTEGER NOT NULL,
    displayed_rank INTEGER NOT NULL CHECK(displayed_rank > 0),
    score REAL NOT NULL,
    selection_source TEXT NOT NULL CHECK(
        selection_source IN ('base_ranker', 'personal_ranker', 'exploration')
    ),
    created_at TEXT NOT NULL,
    FOREIGN KEY(recommendation_run_id) REFERENCES recommendation_runs(id),
    FOREIGN KEY(paper_id) REFERENCES papers(id),
    UNIQUE(recommendation_run_id, paper_id),
    UNIQUE(recommendation_run_id, displayed_rank)
);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    impression_id INTEGER NOT NULL,
    value TEXT NOT NULL CHECK(value IN ('interested', 'not_interested')),
    created_at TEXT NOT NULL,
    FOREIGN KEY(impression_id) REFERENCES impressions(id)
);

CREATE INDEX idx_papers_arxiv_id ON papers(arxiv_id);
CREATE INDEX idx_recommendation_runs_started_at
    ON recommendation_runs(run_started_at);
CREATE INDEX idx_impressions_run_id ON impressions(recommendation_run_id);
CREATE INDEX idx_impressions_paper_id ON impressions(paper_id);
CREATE INDEX idx_feedback_impression_id ON feedback(impression_id);
CREATE INDEX idx_feedback_created_at ON feedback(created_at);
