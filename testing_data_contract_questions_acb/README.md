# Golden-set data-contract question collection

Real-LLM run of `agents.service.generate_data_contract_questions_for_batch` — the exact
function `agents/graph.py`'s `generate_data_contract_questions` node and `make questions` both
call, nothing mocked — question-generation **stage 2 of 2**. Stage 1
(`testing_arch_questions_acb/`) identifies which components and data contracts exist; this
stage takes a fixed list of already-identified contracts and drafts the full ODCS
v3.0.0-completeness question set for each one.

Splitting the two stages apart fixed a real, observed failure mode: a single combined pass
regularly under-covered data contracts even when component/architecture coverage was thorough
— see `doc/cost_analysis.md` for the numbers that motivated the split.

## The cases

Each `golden_set/<case>/` directory holds `transcript.txt` and `contracts.json` — a
hand-authored `mentioned_data_contracts` fixture (`[{"name", "producer", "consumer",
"action"}]`) standing in for what a real `testing_arch_questions_acb` run would have
identified for that same transcript, so each case is self-contained (no live stage-1 call
needed to exercise stage 2).

| Case | Contracts | What it tests |
|---|---|---|
| `01_new_contract_unclear_status` | 1, `action: unknown` | A contract whose lifecycle action wasn't resolved by stage 1 — stage 2 must still draft full identity/schema/quality questions for it, plus (per `data_contract_questions.jinja` PHASE 3/Phase on unresolved action) surface the missing action itself. |
| `02_new_contract_partial_schema` | 1, `action: new` | Some schema facts are already known (cart items, customer ID, computed price) but their types/formats aren't — tests PHASE 9 (partial information): ask only for what's missing, not the whole schema again. |
| `03_multiple_contracts_freshness_quality` | 2 | Two contracts at once, one with explicit-but-uncertain freshness/retention language ("near real-time, not sure how many seconds", "a specific number of days but I forget") — tests quality/SLA depth *and* that both contracts get real, separate coverage, not one contract crowding out the other. |
| `04_modified_contract_versioning` | 1, `action: modified` | A contract gaining one new field — tests PHASE 4/5 (versioning: previous/new version, reason, compatibility, affected consumers) specifically. |

## How it works

Same two-tier split as `testing_arch_questions_acb/`: free, deterministic checks (structural
integrity, plus every drafted question's `target` matching a contract actually in
`contracts.json` — this stage must never invent one) gate the test and set `result.json`'s
`passed`. Only if those pass does the Critic run — scoring 0-100 whether answering every
question would leave each listed contract with a complete ODCS spec, recorded for information
only; see `testing_arch_questions_acb/README.md` for why a probabilistic judge doesn't gate
this suite either.

## Run

```bash
make test-data-contract-questions-acb
```

## Output

Everything lands under this directory's own `output/` (gitignored). Each case's full drafted
result lives at `output/ingestion_date=golden-<name>/data_contract_questions/<name>.json`.
`output/result.json` is the at-a-glance summary: `{"name": ..., "question_count": ...,
"contract_count": ..., "passed": ..., "score": ..., "reason": ..., "input_tokens": ...,
"output_tokens": ..., "estimated_euro_cost": ...}`, printed to the terminal via pytest's
terminal-summary hook, visible without needing `-s`.
