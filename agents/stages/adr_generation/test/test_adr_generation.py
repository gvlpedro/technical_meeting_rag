"""Fast, deterministic tests for the ADR-generation stage's own helpers, in
`agents/stages/adr_generation/service.py`. No LLM is involved anywhere in this file.
Diagram extraction, color-stripping, and the placeholder-leak check are all pure string
logic. The actual ADR-writing LLM call is tested for real in `agents/stages/adr_generation/testing/`'s golden
set instead."""

import pytest

from agents.stages.adr_generation.prompts import build_adr_generation_prompt
from agents.stages.adr_generation.service import (
    adr_drops_suggest_info_content,
    adr_has_placeholder_leak,
    own_previous_architecture_diagram,
    previous_target_architecture_diagram,
)

_SAMPLE_ADR = """# ADR — Introduce Payment Gateway

## 1. ADR

### Context

Payment Gateway is a new component.

## 2. Previous Architecture

The previous architecture is not described by the transcript and clarifications.

## 3. Target Architecture

```mermaid
flowchart LR
    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]
```

## 4. Affected Components

| Component | Change | Description |
|---|---|---|
| Payment Gateway | **NEW** | Processes card payments. |

## 5. Affected Data Contracts

No data contract changes were confirmed by the transcript and clarifications for this change.
"""


def test_previous_target_architecture_diagram_extracts_the_mermaid_block():
    diagram = previous_target_architecture_diagram(_SAMPLE_ADR)
    assert diagram == (
        "flowchart LR\n    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]"
    )


def test_previous_target_architecture_diagram_is_empty_when_there_is_no_previous_adr():
    assert previous_target_architecture_diagram(None) == ""
    assert previous_target_architecture_diagram("") == ""


def test_previous_target_architecture_diagram_is_empty_when_the_section_has_no_diagram():
    adr_without_diagram = _SAMPLE_ADR.replace(
        "```mermaid\n"
        "flowchart LR\n"
        "    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]\n"
        "```",
        "The previous architecture is not described by the transcript and clarifications.",
    )
    assert previous_target_architecture_diagram(adr_without_diagram) == ""


_SAMPLE_ADR_WITH_PREVIOUS_DIAGRAM = """# ADR — Enrich Payment Gateway

## 1. ADR

### Context

Payment Gateway already exists; this change adds fraud scoring.

## 2. Previous Architecture

```mermaid
flowchart LR
    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]
```

## 3. Target Architecture

```mermaid
flowchart LR
    classDef nodeNew fill:#34d399,stroke:#047857,color:#022c22
    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]
    PaymentGateway --> FraudScorer[Fraud Scorer]
    class FraudScorer nodeNew
```

**Legend:** \U0001f7e2 New

## 4. Affected Components

| Component | Change | Description |
|---|---|---|
| Fraud Scorer | **NEW** | Scores transactions for fraud risk. |
"""


def test_own_previous_architecture_diagram_extracts_this_drafts_own_section_2():
    diagram = own_previous_architecture_diagram(_SAMPLE_ADR_WITH_PREVIOUS_DIAGRAM)
    assert diagram == (
        "flowchart LR\n    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]"
    )
    # This must stay distinct from §3. Regenerating must never promote this draft's own
    # target into "previous".
    assert diagram != previous_target_architecture_diagram(_SAMPLE_ADR_WITH_PREVIOUS_DIAGRAM)


def test_own_previous_architecture_diagram_is_empty_when_there_is_no_draft():
    assert own_previous_architecture_diagram(None) == ""
    assert own_previous_architecture_diagram("") == ""


def test_previous_target_architecture_diagram_strips_color_classes_and_legend():
    # §3 of this sample is colored. It has a NEW node, its classDef, and a legend line.
    # When this diagram becomes a LATER ADR's §2 Previous Architecture, it must never carry
    # that coloring forward. §2 is always a plain, colorless snapshot. This holds no matter
    # how the ADR that drew this diagram styled it.
    diagram = previous_target_architecture_diagram(_SAMPLE_ADR_WITH_PREVIOUS_DIAGRAM)
    assert "classDef" not in diagram
    assert "class FraudScorer" not in diagram
    assert "Legend" not in diagram
    assert diagram == (
        "flowchart LR\n"
        "    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]\n"
        "    PaymentGateway --> FraudScorer[Fraud Scorer]"
    )


def test_own_previous_architecture_diagram_is_empty_when_section_2_has_no_diagram():
    assert own_previous_architecture_diagram(_SAMPLE_ADR) == ""


def test_adr_has_placeholder_leak_is_false_for_a_clean_document():
    assert adr_has_placeholder_leak(_SAMPLE_ADR) is False


@pytest.mark.parametrize(
    "leak",
    [
        "<!-- fill this in -->",
        "TBD",
        "TODO",
        "| |",
        "- [ ] unchecked",
        "[Describe the alternative]",
    ],
)
def test_adr_has_placeholder_leak_catches_each_known_pattern(leak: str):
    assert adr_has_placeholder_leak(f"{_SAMPLE_ADR}\n{leak}\n") is True


# --- Prompt content (agents/stages/adr_generation/prompts.py) --------------------------------


def test_adr_generation_prompt_includes_transcript_and_answered_clarifications():
    messages = build_adr_generation_prompt(
        "TRANSCRIPT TEXT", [{"question": "Is X new?", "answer": "Yes, X is new."}]
    )
    content = messages[0]["content"]

    assert "TRANSCRIPT TEXT" in content
    assert "Is X new?" in content
    assert "Yes, X is new." in content


def test_adr_generation_prompt_renders_unanswered_clarification_as_not_answered():
    messages = build_adr_generation_prompt(
        "TRANSCRIPT TEXT", [{"question": "What tokenizes tweets?", "answer": None}]
    )
    content = messages[0]["content"]

    assert "What tokenizes tweets?" in content
    assert "(not answered)" in content


def test_adr_generation_prompt_enforces_component_inclusion_and_no_placeholder_rules():
    content = build_adr_generation_prompt("TRANSCRIPT", [])[0]["content"]

    assert "do not" in content.lower() and "invent a fifth" in content.lower()
    assert "zero" in content.lower() and "placeholder" in content.lower()


def test_adr_generation_prompt_includes_identified_components_and_contracts():
    """Regression test for a real bug: a from-scratch transcript ("one backend serving ... to
    our frontend") was correctly classified by identification as `frontend: new`, `backend:
    new`, but the ADR generator — never told about that classification — independently
    re-derived component status from the bare transcript/clarifications and excluded Frontend
    from §4, so it rendered uncolored in the diagram. `mentioned_components`/
    `mentioned_data_contracts` must now reach the rendered prompt."""
    messages = build_adr_generation_prompt(
        "TRANSCRIPT",
        [],
        mentioned_components=[{"name": "frontend", "status": "new"}, {"name": "backend", "status": "new"}],
        mentioned_data_contracts=[
            {"name": "unknown", "producer": "frontend", "consumer": "backend", "action": "new"}
        ],
    )
    content = messages[0]["content"]

    assert "frontend: new" in content
    assert "backend: new" in content
    assert "frontend -> backend (new)" in content


def test_adr_generation_prompt_renders_none_identified_when_nothing_passed():
    content = build_adr_generation_prompt("TRANSCRIPT", [])[0]["content"]

    assert content.count("(none identified)") == 2


# --- adr_drops_suggest_info_content (agents/stages/adr_generation/service.py) -----------------


def test_adr_drops_suggest_info_content_is_true_when_marker_answered_but_no_suggestion_appears():
    """Regression test for a real, reproduced bug: a clarification answered `[SUGGEST INFO]`
    (e.g. "how do the frontend and backend interact?") sometimes got silently dropped — no
    `LLM SUGGESTION:` text anywhere, no diagram edge — leaving the two components disconnected
    in the generated ADR's own diagram, and therefore in Gold's live architecture diagram too."""
    qa_pairs = [{"question": "How do X and Y interact?", "answer": "[SUGGEST INFO]"}]
    document = "# ADR\n\nX and Y are both introduced. No mention of how they interact."

    assert adr_drops_suggest_info_content(document, qa_pairs) is True


def test_adr_drops_suggest_info_content_is_false_when_a_suggestion_is_present():
    qa_pairs = [{"question": "How do X and Y interact?", "answer": "[SUGGEST INFO]"}]
    document = "# ADR\n\nLLM SUGGESTION: X calls Y over a REST API."

    assert adr_drops_suggest_info_content(document, qa_pairs) is False


def test_adr_drops_suggest_info_content_is_false_when_no_clarification_was_suggest_info():
    qa_pairs = [{"question": "What is X?", "answer": "X is a new component."}]
    document = "# ADR\n\nX is a new component. No suggestions were needed here."

    assert adr_drops_suggest_info_content(document, qa_pairs) is False


def test_adr_drops_suggest_info_content_is_false_for_no_clarifications_at_all():
    assert adr_drops_suggest_info_content("# ADR\n\nSome content.", []) is False
