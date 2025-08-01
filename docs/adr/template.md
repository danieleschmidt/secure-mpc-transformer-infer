# ADR Template

Use this template for creating new Architecture Decision Records (ADRs).

## ADR Format

```markdown
# ADR-XXXX: [Short descriptive title]

**Status:** [Proposed | Accepted | Deprecated | Superseded]
**Date:** YYYY-MM-DD
**Authors:** [Name(s)]
**Reviewers:** [Name(s)]
**Related ADRs:** [ADR-YYYY, ADR-ZZZZ]

## Context

Describe the context and problem statement that necessitates this decision. Include:
- Business requirements or constraints
- Technical requirements or constraints
- Current state and limitations
- Forces at play (technical, political, social, project local)

## Decision

State the architecture decision and provide clear reasoning. Include:
- What we decided
- Why we decided it
- How it addresses the context and constraints

## Alternatives Considered

Document the alternative options considered and why they were rejected:

### Alternative 1: [Name]
- **Description:** Brief description
- **Pros:** Benefits of this approach
- **Cons:** Drawbacks and limitations
- **Rejection Reason:** Why this wasn't chosen

### Alternative 2: [Name]
- **Description:** Brief description
- **Pros:** Benefits of this approach
- **Cons:** Drawbacks and limitations
- **Rejection Reason:** Why this wasn't chosen

## Consequences

Document the consequences of this decision:

### Positive Consequences
- Benefit 1
- Benefit 2
- Benefit 3

### Negative Consequences
- Drawback 1
- Drawback 2
- Drawback 3

### Neutral Consequences
- Impact 1
- Impact 2

## Implementation

Describe how this decision will be implemented:
- Implementation plan
- Timeline and milestones
- Required resources
- Success criteria
- Monitoring and measurement

## References

- [Link to related documentation]
- [Link to research papers]
- [Link to external resources]
- [Link to discussions or RFCs]
```

## ADR Numbering

ADRs should be numbered sequentially starting from ADR-0001. Use the format `ADR-XXXX` where XXXX is a four-digit number with leading zeros.

## ADR Status Lifecycle

- **Proposed:** The ADR is under discussion and review
- **Accepted:** The ADR has been approved and should be implemented
- **Deprecated:** The ADR is no longer relevant but kept for historical purposes
- **Superseded:** The ADR has been replaced by a newer ADR

## File Naming Convention

ADR files should be named using the format: `ADR-XXXX-short-descriptive-title.md`

Examples:
- `ADR-0001-gpu-acceleration-framework.md`
- `ADR-0002-mpc-protocol-selection.md`
- `ADR-0003-container-orchestration-platform.md`

## Review Process

1. Create the ADR using this template
2. Submit as a pull request for review
3. Address feedback and iterate
4. Once approved, merge and update status to "Accepted"
5. Implement the decision as described

## Index

Maintain an index of all ADRs in `docs/adr/index.md` for easy navigation and reference.