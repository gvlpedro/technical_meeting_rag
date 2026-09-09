# Golden-set question collection

Real-LLM run of `agents.service.generate_architecture_questions_for_batch` — the exact production
function `agents/graph.py`'s `generate_architecture_questions` node and `make questions` both call,
nothing mocked — against ten synthetic transcripts of increasing complexity, from one
clean, single-component description (`01_single_component.txt`) to a deliberately
chaotic, ambiguously-named ten-plus-component mess (`10_very_complex_unclear.txt`).

## How it works

Two independent things happen per case, and only one of them can fail the test.

Free, deterministic checks run first — mechanical integrity checks on the collection
itself, not a quality judgment: a structural check (too few questions, duplicate ids,
duplicate question text, or a "question" with no `?`) and a grounding check (every
`mentioned_components` entry the Actor claims must appear in the transcript verbatim —
catches a hallucinated or paraphrased component name with a plain string match). A case
failing one of these is a real, mechanical bug worth looking at (the golden set has
caught genuine ones — an invented component name on a deliberately ambiguous transcript,
for one). This outcome is `result.json`'s `passed` field, and it's the only thing the
test actually asserts on.

Only if those pass does the **Critic** run — a second, independent LLM call scoring 0-100
whether a human answering every drafted question would leave the architecture change
fully and accurately documented, plus a one-paragraph `reason`. Its prompt lives in this
directory's own `critic_prompt.jinja` (rendered via `_build_evaluation_prompt`), not inline
in the test module — the same convention `prompts/*.jinja` uses for
production prompts, just scoped to this test suite. It's written against general
principles (completeness against what the transcript actually raises, no padding), not
against `architecture_questions.jinja`'s own internal structure — an earlier version of
this rubric was tied to a specific, heavily-specified prompt and went stale the moment the
prompt got rewritten. `score`/`reason` are recorded for information only; a probabilistic
judge deciding pass/fail is exactly what made this suite flaky before, so only the
deterministic checks above gate it now.

## Run

```bash
make test-questions-acb

# stricter floor on question count:
MIN_QUESTIONS=15 uv run pytest testing_arch_questions_acb/ -v
```

## Output

Everything lands under this directory's own `output/` (gitignored), never the repo-root
one — `conftest.py` redirects `settings.output_dir` for the run, so this can't pollute
real ingestion data. Each transcript gets the **full** drafted result at its usual
question file, `output/ingestion_date=golden-<name>/questions/<name>.json`,
byte-for-byte what a real `generate_architecture_questions` run would write —
`{"mentioned_components": [{"name": ..., "status": ...}, ...], "questions": [{"id": ...,
"scope": ..., "target": ..., "requirement": ..., "question": ...}, ...]}`. That's where
to go to actually read the drafted questions.

Once the run finishes, `conftest.py` separately writes `output/result.json` — a
**summary**, one entry per transcript in golden-set order, not a second copy of the full
content above: `{"name": ..., "question_count": ..., "passed": ..., "score": ...,
"reason": ..., "input_tokens": ..., "output_tokens": ..., "estimated_euro_cost": ...}`.
`passed` reflects the structural/grounding checks only (see **How it works**, above);
`score`/`reason` are the Critic's own judgment, recorded even though they don't gate the
test — treat a low score as a cue to go read that case's full question file, not as
ground truth on its own. The token/cost fields cover every real LLM call that case made
(Actor + Critic, when the Critic ran), metered by wrapping `litellm.acompletion` for the
duration of the test and pricing each response with `litellm.completion_cost`; the EUR
figure is a fixed-rate (`USD_TO_EUR`) conversion, not a live FX lookup — an
order-of-magnitude estimate, not an invoice. `result.json`'s content is also printed to
the terminal via pytest's terminal-summary hook, visible without needing `-s`.
