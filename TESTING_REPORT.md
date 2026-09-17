# Testing Report

Snapshot of `testing_arch_questions_acb/`'s golden-set evaluation for the **current**
implementation of `prompts/architecture_questions.jinja` — the architecture-questions
generation stage (`agents/service.py::generate_architecture_questions_for_batch`, the same
function `agents/graph.py`'s `generate_architecture_questions` node and `make questions` call).

Keep this file updated whenever that prompt changes materially — re-run the comparison
(**How to reproduce**, below) and replace the table so it always reflects "current
implementation vs. the version before the last material change."

## 2026-09-16 — Selectivity refinement

**Baseline:** the last commit before this session's prompt work (`1ef587a`).

| Metric | Before | After |
|---|---|---|
| Average questions per transcript | 32.2 | 5.9 |
| Independent Critic score (0-100) | 36.6 | 68.4 |
| Cases passing the structural check | 8/10 | 10/10 |

## How to reproduce

```bash
# 1. Snapshot the prompt version you want as the "before" baseline, e.g. from git:
git show <baseline-ref>:prompts/architecture_questions.jinja > /tmp/before.jinja

# 2. Run the golden set against the baseline
cp prompts/architecture_questions.jinja /tmp/current.jinja
cp /tmp/before.jinja prompts/architecture_questions.jinja
make test-arch-questions-acb
cp testing_arch_questions_acb/output/result.json /tmp/result_before.json

# 3. Restore the current prompt and run again
cp /tmp/current.jinja prompts/architecture_questions.jinja
make test-arch-questions-acb
cp testing_arch_questions_acb/output/result.json /tmp/result_after.json

# 4. Compare (question_count, score, passed, cost) per case and in aggregate — see
#    testing_arch_questions_acb/README.md for what each result.json field means.
```

`make test-arch-questions-acb` runs 10 real transcripts through 1-2 real LLM calls each
(Actor, and the Critic when the Actor's output passes the structural check) — each full run
costs real money and several minutes; it is not part of the fast `make test` suite.
