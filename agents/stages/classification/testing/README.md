# Golden-set clarification-question classifier

Real-LLM run of `agents.prompts.build_classification_prompt` — the exact prompt builder
`agents/graph.py`'s `classify_questions` node calls — against six hand-authored cases, each
guarding one specific failure mode.

This is the smallest, least-specified prompt in the pipeline (`prompts/question_classifier.
jinja`, ~15 lines) but the single decision that gates every clarification question this graph
run will ever surface to a human: `"answered"` silently drops a question, `"needs_clarification"`/
`"unknown"` keep it. A real production bug traced back here — a healthy, well-drafted question
set from `architecture_questions.jinja` collapsing to 1-2 surfaced questions, because the
classifier over-inferred `"answered"` from confident-sounding-but-non-committal prose (e.g.
reading "unchanged" into a transcript that never actually said so). This suite exists so that
class of bug is caught before it ships, not after a user reports it.

## The cases

| Case | Guards against |
|---|---|
| `01_explicit_new_component` | Positive control — an explicit "brand new" statement must still be read as answered, not swallowed by an over-conservative fix. |
| `02_confident_prose_unresolved_status` | The trap itself: fluent, confident prose describing an interaction is not a stated lifecycle status. |
| `03_future_option_vs_decision` | A described-but-explicitly-undecided future option IS an explicit statement about decision status — must be answered, not left open. |
| `04_partial_disclosure` | A partially-known fact (field exists, type doesn't) must split correctly — answered for what's stated, open for what isn't. |
| `05_genuinely_unknowable` | Distinguishes `"unknown"` (no one in the meeting could know) from `"needs_clarification"` (someone could, if asked). |
| `06_original_bug_repro` | Pins the actual reported bug down permanently — the exact transcript that used to wrongly classify all three components' status as `"unchanged"`. |

## How it works

Unlike this repo's other golden sets, there is no probabilistic Critic here — correctness has a
real, hand-authored ground truth per question (`expected` in each case's `case.json`), and that
IS what gates the test. Each expectation is additive and optional per question:

* `acceptable_statuses` — the classifier's own `status` must be one of these.
* `required_answer_substrings_any_of` — when answered, `answer` must contain at least one
  (case-insensitive) — not all, since exact wording varies run to run.
* `forbidden_answer_substrings` — when answered, `answer` must contain none of these. This is
  what pins down a wrong inference (e.g. "unchanged") even when "answered" itself is acceptable.

Runs at `temperature=0` (+ `reasoning_effort="none"`), the same as production's own
`classify_questions` call — this suite is exactly what motivated fixing that call's
temperature in the first place.

## Run

```bash
make test-classifier-acb
```

## Output

`agents/stages/classification/testing/output/result.json` — one entry per case: `{name, passed, failures,
input_tokens, output_tokens, estimated_euro_cost}`. `failures` is a list of human-readable
strings when `passed` is false, empty otherwise. Printed to the terminal via pytest's own
terminal-summary hook, visible without needing `-s`.
