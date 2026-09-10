from agents.prompts import (
    build_adr_generation_prompt,
    build_architecture_question_generation_prompt,
    build_data_contract_question_generation_prompt,
)


def test_architecture_question_prompt_with_no_architecture_says_none_known():
    messages = build_architecture_question_generation_prompt("TEMPLATE", "TRANSCRIPT", architecture_diagram="")
    content = messages[0]["content"]

    assert "TEMPLATE" in content
    assert "TRANSCRIPT" in content
    assert "No existing architecture was provided." in content
    assert "flowchart" not in content


def test_architecture_question_prompt_with_architecture_includes_the_diagram_verbatim():
    diagram = "flowchart LR\n  thunder[Thunder] --> phoenix[Phoenix]"
    messages = build_architecture_question_generation_prompt(
        "TEMPLATE", "TRANSCRIPT", architecture_diagram=diagram
    )
    content = messages[0]["content"]

    assert diagram in content
    assert "No existing architecture was provided." not in content


def test_architecture_question_prompt_identifies_but_does_not_specify_data_contracts():
    content = build_architecture_question_generation_prompt("TEMPLATE", "TRANSCRIPT", architecture_diagram="")[
        0
    ]["content"]

    # stage 1 identifies contracts (name/producer/consumer/action)...
    assert "mentioned_data_contracts" in content
    # ...but explicitly defers full ODCS depth to the next stage
    assert "next stage" in content.lower()


def test_data_contract_question_prompt_covers_odcs_depth_and_restrains_padding():
    content = build_data_contract_question_generation_prompt("REQUIREMENTS", "TRANSCRIPT", [])[0]["content"]

    # the ODCS-shaped categories a data-contract question should be able to cover: versioning
    # (previous/new version, breaking changes, affected consumers), schema depth (required/
    # optional, constraints), and quality (freshness)
    for expected in (
        "Previous contract version",
        "New contract version",
        "break-change",
        "forward-update",
        "Affected consumers",
        "Required/optional",
        "Constraints",
        "Freshness",
        "duplicate",
    ):
        assert expected in content

    # no denylist of governance/bookkeeping fields in this prompt — instead a general
    # restraint: don't ask about an ODCS property just because ODCS has it
    assert "Do not ask for generic ODCS properties merely because they exist" in content


def test_data_contract_question_prompt_includes_identified_contracts():
    messages = build_data_contract_question_generation_prompt(
        "REQUIREMENTS",
        "TRANSCRIPT",
        [
            {
                "name": "processed-event",
                "producer": "Event Processor",
                "consumer": "Analytics",
                "action": "forward-update",
            }
        ],
    )
    content = messages[0]["content"]

    assert "processed-event" in content
    assert "Event Processor" in content
    assert "Analytics" in content


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
