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
    fix_diagram_class_references,
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


# --- fix_diagram_class_references (agents/stages/adr_generation/service.py) -------------------


def _adr_with_target_architecture(diagram_body: str, table_rows: str, *, include_section_2: bool = False) -> str:
    """Minimal realistic document shape `fix_diagram_class_references` actually keys off:
    `## 3. Target Architecture`'s own fence, followed eventually by `## 4. Affected
    Components`'s own table, followed by `## 5. ...` as the table's own end boundary."""
    section_2 = (
        "## 2. Previous Architecture\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        '    n0["`Frontend v1`"]\n'
        "```\n\n"
        if include_section_2
        else ""
    )
    return (
        f"{section_2}"
        "## 3. Target Architecture\n\n"
        "```mermaid\n"
        f"{diagram_body}\n"
        "```\n\n"
        "**Legend:** 🟢 New · 🟠 Modified · 🔴 Removed\n\n"
        "## 4. Affected Components\n\n"
        "| Component | Change | Description |\n"
        "|-----------|--------|-------------|\n"
        f"{table_rows}\n"
        "## 5. Affected Data Contracts\n\n"
        "No data contract changes were confirmed by the transcript and clarifications for this change.\n"
    )


def test_fix_diagram_class_references_colors_a_component_with_no_class_line_at_all():
    """Regression test for a real, reported bug — a third real shape of the same underlying
    failure, distinct from the two below: §4 correctly says `Frontend` is `**MODIFIED**`, but §3
    has no `class` line for its node at all, and no `classDef nodeModified` either. The model
    simply never wrote one, despite the fact being right there in its own §4 table."""
    document = _adr_with_target_architecture(
        diagram_body=(
            '    n0["`Frontend v1`"]\n'
            '    n1["`Backend v1`"]\n'
            '    n3["`Auth Service v1`"]\n'
            "    n0 --> n3\n"
            "    classDef nodeNew fill:#34d399,stroke:#047857,color:#022c22\n"
            "    class n3 nodeNew"
        ),
        table_rows=(
            "| Auth Service | **NEW** | New component owning signup and login |\n"
            "| Frontend | **MODIFIED** | Now calls Auth Service directly |\n"
        ),
    )
    fixed = fix_diagram_class_references(document)
    assert "classDef nodeModified fill:#fb923c,stroke:#c2410c,color:#431407" in fixed
    assert "class n0 nodeModified" in fixed
    assert "class n3 nodeNew" in fixed  # the already-correct line for Auth Service survives


def test_fix_diagram_class_references_ignores_a_class_line_naming_a_label_instead_of_an_id():
    """Two real, reported variants of a second failure mode — the model named a node by its own
    label instead of its id (`class ia-service nodeNew` / `class frontend nodeModified`, and
    separately `class backend v1 nodeModified`, the label WITH its version). This function no
    longer tries to read or repair those lines at all — it rebuilds §3's styling from §4 from
    scratch, so whatever shape the model's own mistake took here is simply discarded."""
    document = _adr_with_target_architecture(
        diagram_body=(
            '    n0["`frontend v1`"]\n'
            '    n1["`backend v1`"]\n'
            '    n2["`ia-service`"]\n'
            "    n0 --> n1\n"
            "    n0 --> n2\n"
            "    classDef nodeNew fill:#34d399,stroke:#047857,color:#022c22\n"
            "    class ia-service nodeNew\n"
            "    class backend v1 nodeModified"
        ),
        table_rows=(
            "| ia-service | **NEW** | A new chat service |\n"
            "| backend | **MODIFIED** | Now calls ia-service |\n"
        ),
    )
    fixed = fix_diagram_class_references(document)
    assert "class n2 nodeNew" in fixed
    assert "class n1 nodeModified" in fixed
    assert "class ia-service" not in fixed
    assert "class backend v1" not in fixed


def test_fix_diagram_class_references_fixes_a_classdef_declared_for_the_wrong_class_only():
    """A `class` line pointing at a real node id is still worth nothing if NEITHER
    `classDef nodeModified ...` nor `classDef nodeNew ...` was ever declared — confirmed live.
    Since this function rebuilds styling from §4 rather than reading what the model wrote, the
    missing-`classDef` shape of the bug collapses into the same fix as every other shape."""
    document = _adr_with_target_architecture(
        diagram_body=(
            '    n0["`Frontend v2`"]\n'
            '    n1["`Backend v2`"]\n'
            '    n1 --> n5["`Order Placed`"]'
        ),
        table_rows=(
            "| Backend | **MODIFIED** | Extended to support checkout |\n"
            "| Order Placed | **NEW** | Event published on checkout |\n"
        ),
    )
    fixed = fix_diagram_class_references(document)
    assert "classDef nodeModified fill:#fb923c,stroke:#c2410c,color:#431407" in fixed
    assert "classDef nodeNew fill:#34d399,stroke:#047857,color:#022c22" in fixed
    assert "class n1 nodeModified" in fixed
    assert "class n5 nodeNew" in fixed


def test_fix_diagram_class_references_groups_multiple_nodes_of_the_same_class_on_one_line():
    """The prompt's own example groups same-status nodes on one `class` line
    (`class CDS,RNS nodeNew`) — a rebuilt diagram must do the same, not one `class` line per
    node."""
    document = _adr_with_target_architecture(
        diagram_body=(
            '    n0["`Checkout Service v1`"]\n'
            '    n1["`Fraud Scorer v1`"]\n'
            '    n2["`Notifications v1`"]'
        ),
        table_rows=(
            "| Fraud Scorer | **NEW** | Scores checkout risk |\n"
            "| Notifications | **NEW** | Sends checkout alerts |\n"
        ),
    )
    fixed = fix_diagram_class_references(document)
    assert "class n1,n2 nodeNew" in fixed
    assert fixed.count("classDef nodeNew") == 1


def test_fix_diagram_class_references_colors_a_removed_node():
    document = _adr_with_target_architecture(
        diagram_body='    n0["`Frontend v1`"]\n    n1["`Legacy Cache v3`"]',
        table_rows="| Legacy Cache | **REMOVED** | Decommissioned |\n",
    )
    fixed = fix_diagram_class_references(document)
    assert "classDef nodeRemoved fill:#f87171,stroke:#b91c1c,color:#450a0a,stroke-dasharray: 5 5" in fixed
    assert "class n1 nodeRemoved" in fixed


def test_fix_diagram_class_references_handles_plain_bracket_labels_too():
    """The model does not always reproduce Gold's backtick-label style — with no previous
    diagram to reproduce, it writes plain `NodeId[Label]` nodes instead."""
    document = _adr_with_target_architecture(
        diagram_body="    CheckoutService[Checkout Service] --> PaymentGateway[Payment Gateway]",
        table_rows="| Payment Gateway | **NEW** | Processes card payments |\n",
    )
    fixed = fix_diagram_class_references(document)
    assert "class PaymentGateway nodeNew" in fixed


def test_fix_diagram_class_references_skips_a_component_not_resolvable_in_the_diagram():
    """A §4 name that matches no node in §3's own diagram is not something this function can
    safely guess at — it is silently skipped, never invented."""
    document = _adr_with_target_architecture(
        diagram_body='    n0["`Frontend v1`"]',
        table_rows="| Backend | **MODIFIED** | Not actually drawn in this diagram |\n",
    )
    fixed = fix_diagram_class_references(document)
    assert "classDef" not in fixed
    assert "class n0" not in fixed


def test_fix_diagram_class_references_is_a_noop_when_section_4_has_no_qualifying_row():
    document = _adr_with_target_architecture(
        diagram_body='    n0["`Frontend v1`"]',
        table_rows="",
    )
    assert fix_diagram_class_references(document) == document


def test_fix_diagram_class_references_is_a_noop_with_no_section_3_at_all():
    assert fix_diagram_class_references("# ADR\n\nNo architecture sections here at all.") == (
        "# ADR\n\nNo architecture sections here at all."
    )


def test_fix_diagram_class_references_never_touches_section_2():
    """§2 Previous Architecture must never carry a color (a separate, already-enforced rule —
    `strip_diagram_colors`) — this function does not even look at §2's own fence; it only ever
    reads and rewrites §3's."""
    document = _adr_with_target_architecture(
        diagram_body='    n0["`Frontend v1`"]\n    n1["`Backend v2`"]',
        table_rows="| Backend | **MODIFIED** | Extended |\n",
        include_section_2=True,
    )
    fixed = fix_diagram_class_references(document)
    section_2 = fixed[fixed.find("## 2.") : fixed.find("## 3.")]
    assert "classDef" not in section_2
    assert "class " not in section_2


def test_fix_diagram_class_references_works_without_numbered_headings():
    """Regression test for a real, reported bug: a whole generated ADR wrote plain `"##
    Target Architecture"` / `"## Affected Components"` throughout, never the numbered
    `"## 3. ..."` / `"## 4. ..."` form `prompts/adr_generation/generator.jinja`'s OUTPUT
    STRUCTURE specifies — this function used to find no section at all and return the
    document untouched, silently leaving `class Redis nodeNew` / `class Auth Service v2
    nodeModified` broken (the second of those lines breaks the WHOLE diagram's Mermaid parse,
    confirmed live)."""
    document = (
        "## Target Architecture\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        '    n3["`Auth Service v2`"]\n'
        '    n4["`Redis v1`"]\n'
        "    n3 --> n4\n"
        "    classDef nodeNew fill:#34d399,stroke:#047857,color:#022c22\n"
        "    class Redis nodeNew\n"
        "    class Auth Service v2 nodeModified\n"
        "```\n\n"
        "## Affected Components\n\n"
        "| Component | Change | Description |\n"
        "|-----------|--------|-------------|\n"
        "| Redis | **NEW** | Session cache |\n"
        "| Auth Service | **MODIFIED** | Now stores sessions in Redis |\n\n"
        "## Affected Data Contracts\n"
    )
    fixed = fix_diagram_class_references(document)
    assert "classDef nodeModified fill:#fb923c,stroke:#c2410c,color:#431407" in fixed
    assert "class n4 nodeNew" in fixed
    assert "class n3 nodeModified" in fixed
    assert "class Redis nodeNew" not in fixed
    assert "class Auth Service v2" not in fixed
