# ADR 0004: Use SQLite for Local Application Data

- Status: Accepted
- Date: 2026-06-23

## Context

The local application needs durable storage for papers, recommendation runs,
impressions, explicit feedback, and model versions. This data is relational
and must be queryable for chronological evaluation.

PostgreSQL would require users to install and operate a database server. A
specialized vector database would add deployment complexity before the
application needs approximate nearest-neighbor search at scale.

## Decision

Use SQLite as the initial local persistence layer.

Use NumPy-based exact similarity for the initial daily candidate set. Store
embedding metadata in SQLite and choose either SQLite binary storage or
versioned local array files when embedding persistence is implemented.

## Alternatives Considered

- JSON or JSONL files
- PostgreSQL with pgvector
- A dedicated vector database
- A local FAISS index

## Consequences

- The application has no external database service dependency.
- Transactions, schema constraints, and chronological queries are available.
- Database migrations and backups must be handled by the application.
- SQLite is not intended to serve concurrent remote users.
- A separate vector index may be added if the local historical corpus grows
  beyond practical exact-search limits.

## Revisit When

Revisit this decision if the application needs large-scale semantic search,
multiple concurrent users, remote access, or a hosted deployment.
