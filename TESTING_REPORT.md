# Testing Report

## Architecture questions

| Metric | 2026-09-16 | 2026-09-17  |
|---|---|---|
| Average questions per transcript | 5.9 | 6.3 |
| Independent Critic score (0-100) | 68.4 | 77.5 |
| Cases passing the structural check | 10/10 | 10/10 |

## Data contract questions

| Metric | 2026-09-17 |
|---|---|
| Average questions per case | 28.25 (24–35 across the 4 cases) |
| Independent Critic score (0-100) | 83.5 (82–88) |
| Cases passing the structural check | 4/4 |

## Architecture questions — PHASE 6B fix (2026-09-18)

Bug: a component-to-component connection only got a data-contract candidate
(`mentioned_data_contracts`) when that component's lifecycle status was already confirmed
`new`/`modified`/`removed` — but status is itself something the transcript often leaves
unresolved (`status: "unknown"`) until the reviewer answers a question about it, so the rule
never fired for exactly the transcripts that needed it (a from-scratch system with 4
unresolved-status components produced ZERO data contracts and ZERO data-contract questions).
Fixed: the rule now fires for any connection unless BOTH components are CONFIRMED `unchanged`
— `unknown` no longer counts as "safe to skip."

| Metric | Before fix | After fix |
|---|---|---|
| Data contracts identified (repro case: frontend/backend/DB/API-Gateway, all status `unknown`) | 0 | 8 |
| Final pending questions shown to the reviewer (same case) | 1–2 | 11 (10 of them `data_contract`) |
| Golden-set average questions per transcript | 6.3 | 7.6 |
| Golden-set Critic score | 77.5 | 77.7 |
| Golden-set structural pass | 10/10 | 10/10 |

## Classifier golden-set + determinism pass (2026-09-19)

New suite `agents/stages/classification/testing/` (6 cases, `make test-classifier-acb`) — `prompts/
question_classifier.jinja` had no real-LLM benchmark of any kind before this, despite being the
single decision that gates every clarification question the graph will ever surface to a human.
6/6 pass, including a permanent repro of the original "only 1 question" bug (case
`06_original_bug_repro`, which forbids the classifier answering any of the three components'
lifecycle status as `"unchanged"`).

`classify_questions` (agents/graph.py) and `synthesize_document`'s ADR-writing call now both run
at `temperature=0` (+ `reasoning_effort="none"`) — both used to run at the provider default,
and `synthesize_document` visibly flip-flopped pass/fail on the SAME golden-set case across
identical runs with nothing else changed. Fixing `synthesize_document`'s temperature exposed a
real, previously-masked prompt bug: at temperature=0 the model consistently (not just
sometimes) collapsed "## 2. Previous Architecture" to "not described" whenever "## 4. Affected
Components" had no confirmed status tags — conflating two separate gates
(`adr_generator.jinja`'s COMPONENT INCLUSION RULE vs. its §2 instructions), something
higher-temperature sampling had been randomly getting right often enough to hide. Fixed by
making the distinction explicit in the prompt; `synthesize_document` also gained a mechanical
placeholder-leak check (`agents.service.adr_has_placeholder_leak`) with the same
shallow-retry-at-higher-temperature escape hatch the question-generation stages already use.

| Metric | Before | After |
|---|---|---|
| `agents.stages.adr_generation.testing`, 3 consecutive runs, no code change between them | 2/3, 2/3, 2/3 passed (same case failing every time, once temperature was already 0 but before the §2 fix) | 3/3, 3/3, 3/3 |
| `agents.stages.classification.testing` | (suite didn't exist) | 6/6 |

## Architecture-questions decomposition — prototype (2026-09-19)

`prompts/architecture_questions/combined.jinja` is one ~1450-line prompt doing three jobs at once
(identify components/contracts, draft candidate questions, select the final set). Built the
same three jobs as three separate, smaller, independently-callable prompts/functions, **not
wired into `agents/graph.py`** — production still calls the single combined prompt unchanged:

* `prompts/architecture_questions/identification.jinja` + `agents.service.identify_architecture_entities`
  — identification only (~180 lines).
* `prompts/architecture_questions/drafting.jinja` + `agents.service.draft_architecture_questions`
  — a deliberately broad, ungated candidate list, informed by call 1's output.
* `prompts/architecture_questions/selection.jinja` + `agents.service.select_architecture_questions`
  — call 2's candidates filtered/deduped/value-scored down to the final set.

Smoke-tested end-to-end with a real transcript (the same frontend/backend/database/API-Gateway
case from the PHASE 6B fix above): call 1 correctly identified 4 real components and all 8
connection-implied data contracts (plus one spurious entry, "multiple services", worth tightening
before this is trusted further); call 2 drafted 55 broad candidates across every scope; call 3
selected 8 well-targeted final questions covering the actual key ambiguities (is the Gateway
current or future, does it replace the direct frontend-backend path, are multiple services in
scope). All three calls run at `temperature=0` per the same discipline established above.

**Not yet done** (deliberately, pending a decision): a full A/B against the current monolithic
prompt on `agents.stages.architecture_questions.testing`'s 10-case golden set, and any decision to retire the
combined prompt. Until that comparison exists, treat this as an evaluation prototype, not a
replacement — the extra LLM calls (3 vs. 1) have a real latency/cost cost that needs to be
weighed against whatever quality/testability gain the split turns out to have.

## How to reproduce

```bash
# 1. Snapshot the prompt version you want as the "before" baseline, e.g. from git:
git show <baseline-ref>:prompts/architecture_questions/combined.jinja > /tmp/before.jinja

# 2. Run the golden set against the baseline
cp prompts/architecture_questions/combined.jinja /tmp/current.jinja
cp /tmp/before.jinja prompts/architecture_questions/combined.jinja
make test-arch-questions-acb
cp agents/stages/architecture_questions/testing/output/result.json /tmp/result_before.json

# 3. Restore the current prompt and run again
cp /tmp/current.jinja prompts/architecture_questions/combined.jinja
make test-arch-questions-acb
cp agents/stages/architecture_questions/testing/output/result.json /tmp/result_after.json

# 4. Compare (question_count, score, passed, cost) per case and in aggregate — see
#    agents/stages/architecture_questions/testing/README.md for what each result.json field means.
```
