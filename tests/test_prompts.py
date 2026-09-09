from agents.prompts import build_adr_generation_prompt, build_question_generation_prompt


def test_question_generation_prompt_with_no_architecture_says_none_known():
    messages = build_question_generation_prompt("TEMPLATE", "TRANSCRIPT", architecture_diagram="")
    content = messages[0]["content"]

    assert "TEMPLATE" in content
    assert "TRANSCRIPT" in content
    assert "No existing architecture was provided." in content
    assert "flowchart" not in content


def test_question_generation_prompt_with_architecture_includes_the_diagram_verbatim():
    diagram = "flowchart LR\n  thunder[Thunder] --> phoenix[Phoenix]"
    messages = build_question_generation_prompt("TEMPLATE", "TRANSCRIPT", architecture_diagram=diagram)
    content = messages[0]["content"]

    assert diagram in content
    assert "No existing architecture was provided." not in content


def test_question_generation_prompt_covers_data_contract_depth_and_restrains_padding():
    content = build_question_generation_prompt("TEMPLATE", "TRANSCRIPT", architecture_diagram="")[0][
        "content"
    ]

    # the ODCS-shaped categories a data-contract question should be able to cover: versioning
    # (previous/new version, breaking changes, affected consumers), schema depth (required/
    # optional, constraints), and quality (freshness)
    for expected in (
        "Previous contract version",
        "New contract version",
        "breaking change",
        "Affected consumers",
        "Required/optional",
        "Constraints",
        "Freshness",
        "duplicate",
    ):
        assert expected in content

    # no denylist of governance/bookkeeping fields in this version of the prompt — instead a
    # general restraint: don't ask about an ODCS property just because ODCS has it
    assert "Do not ask for generic ODCS properties merely because they exist" in content


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
