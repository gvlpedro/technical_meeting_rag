# Golden-set ADR generation

Real-LLM run of `agents.prompts.build_adr_generation_prompt` — which renders
`prompts/adr_generator.jinja` — against three cases: a real meeting transcript
with a human review pass, and two synthetic cases of increasing complexity. Mirrors
`testing_arch_questions_acb/`'s own structure and philosophy, one stage later in the pipeline:
that project tests turning a transcript into clarification *questions*; this one tests turning
a transcript plus its already-*answered* clarifications into the final ADR document, shaped
like `doc/adr_example.md`.

## The cases

| Case | Transcript | What it tests |
|---|---|---|
| `01_twitter_real_time_delivery` | Real: `real_time_delivery_architecture_at_twitter.en.vtt` | A 36-question factual verification pass describing how the *current* system works — none of the clarifications confirm a lifecycle status for any component, so the strictest expectation applies: the Decision/Affected-Components/Data-Contract sections must all collapse to their "not covered" notes, with no invented change. |
| `02_checkout_loyalty_integration` | Synthetic | Everything resolved cleanly: every component status and the one data contract's version evolution are confirmed. Exercises the full document structure, including a populated ODCS section. |
| `03_multi_region_notification_platform` | Synthetic | Three partial-information patterns at once: an out-of-scope component (confirmed as out of scope, not just silent), an unanswered *detail* question for an otherwise in-scope component, and a data contract whose existence is confirmed but whose schema isn't. |

Each case directory holds `transcript.vtt` (case 01, the real file) or `transcript.txt`
(synthetic cases), `clarifications.json` (`[{"question": ..., "answer": ...}]`, `answer: null`
for a question that was asked but never resolved), `metadata.json`
(`expected_components`/`excluded_components` — used by the deterministic checks below, see
each case's own `description` field for why), and `checklist.txt` — a case-specific,
mostly-deterministic list of ground-truth assertions derived directly from that case's own
`clarifications.json` (e.g. "these exact four components, these exact statuses, zero data
contracts"), fed to the Critic (see below).

## How it works

Two independent things happen per case, and only one of them can fail the test.

Free, deterministic checks run first — mechanical checks on the generated document, not a
quality judgment:

* **No placeholders** — no `<!-- -->` comments, `TBD`/`TODO`, empty table cells, bracketed
  template instructions, or leftover template example text (`Component A`, `contract-a.json`,
  ...) anywhere in the output.
* **Expected components present** — every component a case's clarifications actually confirmed
  a status for must be discussed somewhere in the document.
* **Excluded components not marked affected** — a component whose status was never confirmed
  (an unanswered or explicitly out-of-scope one) must never appear tagged with a change status
  (`**NEW**`/`**MODIFIED**`/`**REMOVED**`/`**UNCHANGED**`) — a bare mention in prose is fine,
  a status tag is exactly the "unknown point" this whole exercise is meant to catch.

This is `result.json`'s `passed` field, and it's the only thing the test actually asserts on.

Only if those pass does the **Critic** run — a second, independent LLM call judging two things:
the same four general criteria (no placeholders, no unknown points for any component, a
crystal-clear implementation description, only established components discussed), *and* every
line of that case's own `checklist.txt`, reported individually as
`checklist_results` (`{item, passed, note}` per line — see `_build_critic_prompt`). A checklist
line tagged `[DETERMINISTIC]` has one factually correct answer and is judged strictly; one
tagged `[JUDGMENT]` allows reasonable latitude in wording. `critic_passes`/`critic_reason`
(which appends any failed checklist items) are recorded for information only — a probabilistic
judge deciding pass/fail is exactly what made `testing_arch_questions_acb` (formerly `testing_questions_acb`) flaky before (see that
project's own README), so only the deterministic checks above gate this suite too.

## Run

```bash
make test-adr-acb
```

## Output

Everything lands under this directory's own `output/` (gitignored). Each case's full generated
ADR is written to `output/<case-name>/adr.md` — that's where to go to actually read what the
model produced. `output/result.json` is the at-a-glance summary across the whole golden set:
`{"name": ..., "passed": ..., "violations": [...], "critic_passes": ..., "critic_reason": ...,
"input_tokens": ..., "output_tokens": ..., "estimated_euro_cost": ...}`, printed to the
terminal via pytest's terminal-summary hook, visible without needing `-s`.
