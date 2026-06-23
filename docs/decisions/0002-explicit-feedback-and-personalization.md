# ADR 0002: Learn From Explicit Binary Feedback

- Status: Accepted
- Date: 2026-06-23

## Context

There is no universal ground truth for whether a research paper is useful to a
specific user. Relevance depends on the user's current interests and goals.
The application therefore needs to collect preference data during normal use
and improve recommendations over time.

Additional actions such as "save for later", "hide", and "already read" do not
provide a clearer training target for the initial daily-paper workflow.

## Decision

Present two explicit feedback choices for a shown paper:

- `interested`
- `not_interested`

Store explicit feedback locally together with the recommendation impression
that produced it. Missing feedback remains unlabeled and must not be treated as
negative feedback.

Personalization will be automatic. Users will not select an ML algorithm.
Initially, the embedding model remains frozen and a lightweight personal
re-ranker may learn from accumulated feedback. More complex adaptation, such
as adapters or LoRA, will be considered only when enough feedback and local
evaluation support it.

## Alternatives Considered

- Asking users to provide numeric relevance scores
- Using implicit clicks as negative or positive labels
- Providing several overlapping feedback actions
- Fine-tuning the complete embedding model after a small number of labels

## Consequences

- The interaction is simple and produces an unambiguous binary target.
- Impression logging is required to interpret feedback and evaluate rankings.
- Position and selection bias must be considered because users label papers
  chosen by the current recommender.
- The system needs controlled exploration to gather feedback outside the
  current model's highest-ranked results.
- Ranker updates must be evaluated before replacing the active ranker.

## Revisit When

Revisit the feedback vocabulary if user research shows that binary preference
does not distinguish important cases, or if the product expands beyond daily
new-paper recommendations.
