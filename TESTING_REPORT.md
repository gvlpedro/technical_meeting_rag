# Testing Report

| Metric | 2026-09-16 | 2026-09-17 | 2026-09-18 | 2026-09-18 v1 | 2026-09-18 v2 | 2026-09-18 v3 |
|---|---|---|---|---------------|---|---------------|
| Architecture Qs — avg per transcript | 5.9 | 6.3 | 7.6 | -             | 6.4 | =             |
| Architecture Qs — Critic score | 68.4/100 | 77.5/100 | 77.7/100 | -             | 79.0/100 | =             |
| Architecture Qs — structural pass | 10/10 | 10/10 | 10/10 | -             | 10/10 | =             |
| Data contract Qs — avg per case | - | 28.25 | - | -             | 28.75 | =             |
| Data contract Qs — Critic score | - | 83.5/100 | - | -             | 82.5/100 | =             |
| Data contract Qs — structural pass | - | 4/4 | - | -             | 4/4 | =             |
| Classifier golden-set — pass rate | - | - | - | 6/6           | 6/6 | =             |
| ADR-generation determinism (3 runs) | - | - | - | 3/3           | - | =             |
| ADR generation golden-set — structural pass | - | - | - | -             | 3/3 | =             |
| ADR generation golden-set — Critic pass (informational) | - | - | - | -             | 1/3 | =             |
| Decomposition prototype — components identified | - | - | - | 4/5           | - | =             |
| Fast suite (`make test`) — pass rate | - | - | - | -             | 130/130 | 134/134       |
| Hybrid search — new unit tests | - | - | - | -             | 6/6 | =             |
| Dedup by entity — new unit tests | - | - | - | -             | - | 4/4           |
| Gold retrieval golden-set — steps run | - | - | - | -             | 5/5 | 5/5           |
| Gold retrieval golden-set — entity_hit rate | - | - | - | -             | 10/10 | 10/10         |
| Gold retrieval golden-set — keyword_hit rate | - | - | - | -             | 10/10 | 10/10         |
