# Architecture Decision Records (ADR) Index

This directory contains all Architecture Decision Records for the Secure MPC Transformer Inference project. ADRs document important architectural decisions, their context, and consequences.

## Active ADRs

| ADR | Title | Status | Date | Authors |
|-----|-------|--------|------|---------|
| [0001](ADR-0001-gpu-acceleration-framework.md) | GPU Acceleration Framework Selection | Accepted | 2024-03-15 | Technical Team |
| [0002](ADR-0002-mpc-protocol-selection.md) | MPC Protocol Selection Criteria | Accepted | 2024-04-02 | Security Team |
| [0003](ADR-0003-container-orchestration.md) | Container Orchestration Platform | Accepted | 2024-04-20 | DevOps Team |
| [0004](ADR-0004-monitoring-observability.md) | Monitoring and Observability Stack | Accepted | 2024-05-10 | SRE Team |
| [0005](ADR-0005-api-design-principles.md) | API Design Principles and Standards | Accepted | 2024-06-01 | API Team |

## Deprecated ADRs

| ADR | Title | Status | Date | Superseded By |
|-----|-------|--------|------|---------------|
| [0000](ADR-0000-initial-architecture.md) | Initial Architecture Proposal | Deprecated | 2024-02-01 | ADR-0001, ADR-0002 |

## ADR Categories

### Security and Privacy
- ADR-0002: MPC Protocol Selection Criteria

### Performance and Scalability  
- ADR-0001: GPU Acceleration Framework Selection

### Infrastructure and Operations
- ADR-0003: Container Orchestration Platform
- ADR-0004: Monitoring and Observability Stack

### API and Integration
- ADR-0005: API Design Principles and Standards

## Creating New ADRs

1. Use the [ADR template](template.md)
2. Follow the [numbering convention](template.md#adr-numbering)
3. Submit via pull request for review
4. Update this index when ADR is accepted

## ADR Review Process

All ADRs must be reviewed by:
- Technical Lead (for technical decisions)
- Security Advisor (for security-related decisions)
- Product Owner (for product/business decisions)

Minimum review period: 5 business days for major architectural decisions.

## Related Documentation

- [Architecture Documentation](../ARCHITECTURE.md)
- [Security Documentation](../security/)
- [Deployment Documentation](../deployment/)
- [Development Guidelines](../DEVELOPMENT.md)