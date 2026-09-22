# Testing Report

| Metric | 2026-09-16 | 2026-09-17 | 2026-09-18 | 2026-09-18 v1 | 2026-09-18 v2 | 2026-09-18 v3 | 2026-09-18 v4 | 2026-09-19 | 2026-09-19 v1 | 2026-09-19 v2 | 2026-09-19 v3 | 2026-09-19 v4 | 2026-09-19 v5 |
|---|---|---|---|---------------|---|---------------|---------------|---|---|---|---|---|---|
| Architecture Qs — avg per transcript | 5.9 | 6.3 | 7.6 | -             | 6.4 | =             | - | - | - | - | - | - | - |
| Architecture Qs — Critic score | 68.4/100 | 77.5/100 | 77.7/100 | -             | 79.0/100 | =             | - | - | - | - | - | - | - |
| Architecture Qs — structural pass | 10/10 | 10/10 | 10/10 | -             | 10/10 | =             | - | - | - | - | - | - | - |
| Data contract Qs — avg per case | - | 28.25 | - | -             | 28.75 | =             | - | - | - | - | - | - | - |
| Data contract Qs — Critic score | - | 83.5/100 | - | -             | 82.5/100 | =             | - | - | - | - | - | - | - |
| Data contract Qs — structural pass | - | 4/4 | - | -             | 4/4 | =             | - | - | - | - | - | - | - |
| Classifier golden-set — pass rate | - | - | - | 6/6           | 6/6 | =             | - | - | - | - | - | - | - |
| ADR-generation determinism (3 runs) | - | - | - | 3/3           | - | =             | - | - | - | - | - | - | - |
| ADR generation golden-set — structural pass | - | - | - | -             | 3/3 | =             | - | - | - | - | - | - | - |
| ADR generation golden-set — Critic pass (informational) | - | - | - | -             | 1/3 | =             | - | - | - | - | - | - | - |
| Decomposition prototype — components identified | - | - | - | 4/5           | - | =             | - | - | - | - | - | - | - |
| Fast suite (`make test`) — pass rate | - | - | - | -             | 130/130 | 134/134       | 139/139 | = | 146/146 | 148/148 | = | 152/152 | 153/153 |
| Hybrid search — new unit tests | - | - | - | -             | 6/6 | =             | - | - | - | - | - | - | - |
| Dedup by entity — new unit tests | - | - | - | -             | - | 4/4           | - | - | - | - | - | - | - |
| Reranking (cross-encoder) — new unit tests | - | - | - | -             | - | -             | 5/5 | - | - | - | - | - | - |
| Gold retrieval golden-set — case count | - | - | - | -             | 5 | =             | = | 10 (doubled) | = | = | = | = | = |
| Gold retrieval golden-set — steps run | - | - | - | -             | 5/5 | 5/5           | n/a (rerank off by default, not exercised) | 10/10 | = | = | = | = | = |
| Gold retrieval golden-set — entity_hit rate | - | - | - | -             | 10/10 | 10/10         | n/a | 20/20 | = | = | = | = | = |
| Gold retrieval golden-set — keyword_hit rate | - | - | - | -             | 10/10 | 10/10         | n/a | 20/20 | = | = | = | = | = |
| Gold retrieval golden-set — checklist violations | - | - | - | -             | 0 | 0             | n/a | 0 | = | = | = | = | = |
| Conversational memory — new unit tests | - | - | - | -             | - | -             | - | - | 7/7 | - | - | - | - |
| Conversational memory — e2e pronoun-resolution check (real app, manual) | - | - | - | -             | - | -             | - | - | pass (2/2 entities correctly disambiguated by history) | - | - | - | - |
| ADR generation — identified-component grounding bugfix, new unit tests | - | - | - | -             | - | -             | - | - | - | 2/2 | - | - | - |
| ADR generation — real-user bug reproduction (real LLM, manual) | - | - | - | -             | - | -             | - | - | - | pass (Frontend now confirmed NEW and colored green, using the exact real transcript+clarifications that originally under-classified it) | - | - | - |
| Question-generation — no Date/Status/Participants questions (real LLM, 2 transcripts, manual) | - | - | - | -             | - | -             | - | - | - | - | pass (0/0 across a from-scratch transcript and one with an explicit date/status stated) | - | - |
| Gold extraction — dependency edges from diagram, new unit test | - | - | - | -             | - | -             | - | - | - | - | - | pass (real LLM: `frontend`/`backend` each correctly listed the other as a dependency, sourced only from the diagram edge, no prose) | - |
| ADR generation — `[SUGGEST INFO]`-drop retry backstop, new unit tests | - | - | - | -             | - | -             | - | - | - | - | - | 4/4 | - |
| ADR generation — retry backstop real-LLM reliability check (manual, 6 runs) | - | - | - | -             | - | -             | - | - | - | - | - | pass (5/6 first attempts dropped the suggestion/edge exactly as before; retry recovered it every single time — 6/6 final results correct) | - |
| Live architecture diagram — real user data repaired (manual, real LLM) | - | - | - | -             | - | -             | - | - | - | - | - | pass (frontend↔backend and backend↔postgres edges now render in the real `/architecture-history` endpoint, regenerated from the user's own stored transcripts/clarifications) | - |
| ADR generation — named-correction retry (vs. blind resample), new unit test | - | - | - | -             | - | -             | - | - | - | - | - | - | 1/1 |
| ADR generation — named-correction reliability check (manual, real LLM, second reported case) | - | - | - | -             | - | -             | - | - | - | - | - | - | pass (6/6 final results correct; every retry now converges in exactly 1 attempt instead of 3-4) |
| Live architecture diagram — second real user ADR repaired (manual, real LLM) | - | - | - | -             | - | -             | - | - | - | - | - | - | pass (frontend↔backend edge now renders for the user's second reported broken ADR) |
