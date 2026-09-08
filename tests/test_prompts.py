from agents.prompts import build_question_generation_prompt


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
