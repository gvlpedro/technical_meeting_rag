# Cost Analysis — Clarification Question Generation

Comparative analysis of LLM model choice for `agents.service.generate_questions_for_batch`,
based on real runs of `testing_questions_acb/`'s golden set (10 transcripts, real OpenAI/
Anthropic API calls, nothing mocked). `score` is the Critic's 0-100 judgment (completeness
minus padding); `price` is the metered cost of the full 10-case suite (Actor + Critic calls).

## Comparative (test suite: `testing_questions_acb`)

| Model | Score (avg.) | Price (10 cases) | Cases passed |
|---|---:|---:|---:|
| `gpt-5.6-sol` | 78.8 | 1.252 € | 10/10 |
| `gpt-5.6-terra` | **81.0** | **0.605 €** | 10/10 |
| `claude-haiku-4-5` | 36.9 | ≈0.26 € (9 cases) | 9/10 |

**Recomiendo `gpt-5.6-terra` como modelo principal**: mismo 10/10 estructural, mejor score que
`sol` (menos padding) y a mitad de coste — no hay razón para pagar `sol`.

**Para el fallback usaría `claude-haiku-4-5`**, no por calidad (36.9 de score, claramente peor)
sino porque es el único modelo de Anthropic que de verdad funciona con `temperature=0` en estas
llamadas: `claude-opus-5` (el fallback configurado hoy) lo rechaza sin escape hatch, así que el
fallback actual está roto de facto — mejor un resultado mediocre que ninguno.

## Per-case detail

### `gpt-5.6-sol`

| Case | Questions | Passed | Score | Cost |
|---|---:|:---:|---:|---:|
| 01_single_component | 60 | ✅ | 72 | 0.097 € |
| 02_two_components_clear | 81 | ✅ | 78 | 0.118 € |
| 03_lifecycle_ambiguity | 92 | ✅ | 82 | 0.126 € |
| 04_data_exchange_unclear_fields | 93 | ✅ | 72 | 0.125 € |
| 05_multiple_interactions_mixed | 91 | ✅ | 78 | 0.127 € |
| 06_sensitive_data_hint | 79 | ✅ | 78 | 0.117 € |
| 07_lifecycle_and_ownership_gaps | 99 | ✅ | 82 | 0.132 € |
| 08_multi_contract_freshness_retention | 98 | ✅ | 82 | 0.132 € |
| 09_complex_mixed_clarity | 113 | ✅ | 82 | 0.145 € |
| 10_very_complex_unclear | 96 | ✅ | 82 | 0.132 € |
| **Total / avg.** | **90.2 avg** | **10/10** | **78.8 avg** | **1.252 €** |

### `gpt-5.6-terra`

| Case | Questions | Passed | Score | Cost |
|---|---:|:---:|---:|---:|
| 01_single_component | 50 | ✅ | 72 | 0.048 € |
| 02_two_components_clear | 68 | ✅ | 88 | 0.058 € |
| 03_lifecycle_ambiguity | 78 | ✅ | 88 | 0.064 € |
| 04_data_exchange_unclear_fields | 63 | ✅ | 72 | 0.056 € |
| 05_multiple_interactions_mixed | 67 | ✅ | 72 | 0.059 € |
| 06_sensitive_data_hint | 51 | ✅ | 88 | 0.051 € |
| 07_lifecycle_and_ownership_gaps | 71 | ✅ | 88 | 0.063 € |
| 08_multi_contract_freshness_retention | 97 | ✅ | 82 | 0.076 € |
| 09_complex_mixed_clarity | 79 | ✅ | 72 | 0.068 € |
| 10_very_complex_unclear | 72 | ✅ | 88 | 0.062 € |
| **Total / avg.** | **69.6 avg** | **10/10** | **81.0 avg** | **0.605 €** |

### `claude-haiku-4-5` (forced Anthropic-only)

| Case | Questions | Passed | Score | Cost |
|---|---:|:---:|---:|---:|
| 01_single_component | 35 | ✅ | 28 | 0.020 € |
| 02_two_components_clear | 52 | ✅ | 32 | 0.025 € |
| 03_lifecycle_ambiguity | 61 | ✅ | 28 | 0.028 € |
| 04_data_exchange_unclear_fields | 52 | ✅ | 42 | 0.025 € |
| 05_multiple_interactions_mixed | 59 | ✅ | 32 | 0.027 € |
| 06_sensitive_data_hint | 48 | ✅ | 72 | 0.024 € |
| 07_lifecycle_and_ownership_gaps | 75 | ✅ | 28 | 0.031 € |
| 08_multi_contract_freshness_retention | 62 | ✅ | 28 | 0.028 € |
| 09_complex_mixed_clarity | 56 | ✅ | 42 | 0.027 € |
| 10_very_complex_unclear | — | ❌ (API error) | — | — |
| **Total / avg. (9 cases)** | **55.6 avg** | **9/10** | **36.9 avg** | **0.235 €** |

## Notes

- `gpt-4o-mini` (original default) was never run on the full suite — the one real data point
  (a minimal single-component transcript) produced 13 questions, all in one scope, well below
  the ">25" bar. That failure is what triggered this whole analysis.
- `reasoning_effort="none"` is required for `sol`/`terra` to accept `temperature=0` at all.
- Untested price-only candidates (`gpt-5.4-mini`, `gpt-5.6-luna`, `claude-sonnet-5`) are not
  included above — no golden-set run exists for them yet. 

## Results

Production default (`app/config.py`) is now `gpt-5.6-terra` / `claude-haiku-4-5`, applying the recommendation above.
