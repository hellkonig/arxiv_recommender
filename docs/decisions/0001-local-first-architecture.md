# ADR 0001: Use a Local-First Architecture

- Status: Accepted
- Date: 2026-06-23

## Context

The recommender is intended for an individual user running the application on a
laptop or desktop computer. Papers, recommendation history, feedback, and
personalization data may contain private information about the user's research
interests.

A hosted service would require account management, centralized storage,
network availability, and operational infrastructure that are not necessary
for the initial product.

## Decision

Run the recommendation workflow, persistence, and personalization locally on
the user's computer.

The initial product will not require a centralized database, hosted inference
service, or shared personalization model. Network access is still required to
retrieve papers and download configured models.

## Alternatives Considered

- A hosted multi-user web application
- A local client backed by a centralized recommendation API
- A shared model updated from feedback collected across users

## Consequences

- User feedback and preference data remain on the user's machine.
- Installation and model execution must work on ordinary local hardware.
- The application must manage local persistence, migrations, and model files.
- Cross-device synchronization and collaborative recommendations are outside
  the initial scope.
- Resource-intensive training techniques may need to be optional.

## Revisit When

Revisit this decision if the product requires multiple users, cross-device
synchronization, shared recommendations, or centrally managed models.
