"""Fast, deterministic tests for `agents/shared.py`. This file holds the cross-stage helpers
that more than one pipeline stage depends on. This test file stays under `tests/`, not under
any `agents/stages/<stage>/test/` folder, because no single stage owns this code."""

from agents.shared import (
    _markdown_section,
    extract_source_line,
    insert_authors_line,
    insert_source_line,
    markdown_heading_pattern,
    mentions_grounded_in_source,
)


def test_insert_source_line_round_trips_through_extract():
    document = "# ADR — Introduce Kafka\n\nSome body text."
    stamped = insert_source_line(document, "meeting.en.vtt")
    assert extract_source_line(stamped) == "meeting.en.vtt"
    assert "Some body text." in stamped


def test_insert_source_line_is_a_noop_for_an_empty_source_component():
    document = "# ADR — Introduce Kafka\n\nSome body text."
    assert insert_source_line(document, "") == document


def test_insert_source_line_composes_with_insert_authors_line():
    """`agents.graph.write_document` chains both stamps together
    (`insert_source_line(insert_authors_line(content, username), source)`) — this is the same
    order, checked here in isolation from the rest of the graph."""
    document = "# ADR — Introduce Kafka\n\nSome body text."
    stamped = insert_source_line(insert_authors_line(document, "alice"), "meeting.en.vtt")
    assert stamped == (
        "# ADR — Introduce Kafka\n\n**Source:** meeting.en.vtt\n\n**Authors:** alice\n\nSome body text."
    )
    assert extract_source_line(stamped) == "meeting.en.vtt"


def test_mentions_grounded_in_source_keeps_only_names_present_in_that_source_text():
    """`mentioned_components` and `mentioned_data_contracts` are drafted once, over the whole
    pooled transcript for an ingestion_date batch. This test shows how `write_document` learns
    which of the batch's mentions actually belong to one specific `source_component`'s row."""
    items = [
        {"name": "Checkout Service", "status": "new"},
        {"name": "Loyalty Service", "status": "new"},
    ]
    source_text = "Today we discussed the Checkout Service and its new payment flow."

    grounded = mentions_grounded_in_source(source_text, items)

    assert grounded == [{"name": "Checkout Service", "status": "new"}]


def test_mentions_grounded_in_source_is_case_insensitive_and_can_match_more_than_one():
    items = [
        {"name": "checkout service", "status": "unchanged"},
        {"name": "Loyalty Service", "status": "new"},
        {"name": "Notification Service", "status": "removed"},
    ]
    source_text = "CHECKOUT SERVICE now calls Loyalty Service before publishing the event."

    grounded = mentions_grounded_in_source(source_text, items)

    assert grounded == [
        {"name": "checkout service", "status": "unchanged"},
        {"name": "Loyalty Service", "status": "new"},
    ]


def test_mentions_grounded_in_source_returns_empty_list_when_nothing_matches():
    items = [{"name": "Checkout Service", "status": "new"}]
    assert mentions_grounded_in_source("A totally unrelated transcript about billing.", items) == []


# --- markdown_heading_pattern / _markdown_section -----------------------------------------------


def test_markdown_heading_pattern_matches_the_numbered_form():
    pattern = markdown_heading_pattern("## 3. Target Architecture")
    assert pattern.search("## 3. Target Architecture\n") is not None


def test_markdown_heading_pattern_also_matches_the_unnumbered_form():
    """Regression test for a real, reported bug: a whole generated ADR wrote plain `"##
    Target Architecture"` / `"## Affected Components"` throughout, never the numbered form
    `prompts/adr_generation/generator.jinja`'s OUTPUT STRUCTURE specifies. Every caller that
    sliced a document by its exact numbered heading silently got an empty section back —
    not just a diagram-coloring miss, but the NEXT ADR's "previous architecture" grounding
    would have been lost entirely, since `previous_target_architecture_diagram` depends on the
    same matching."""
    pattern = markdown_heading_pattern("## 3. Target Architecture")
    assert pattern.search("## Target Architecture\n") is not None


def test_markdown_heading_pattern_does_not_match_a_similar_but_different_heading():
    pattern = markdown_heading_pattern("## 3. Target Architecture")
    assert pattern.search("## Target Architecture Notes\n") is None
    assert pattern.search("## Previous Target Architecture\n") is None


def test_markdown_section_extracts_the_unnumbered_form():
    content = (
        "## Target Architecture\n\nSome diagram text.\n\n## Affected Components\n\nA table.\n"
    )
    section = _markdown_section(content, "## 3. Target Architecture", "## 4. Affected Components")
    assert section == "## Target Architecture\n\nSome diagram text."


def test_markdown_section_still_extracts_the_numbered_form():
    content = (
        "## 3. Target Architecture\n\nSome diagram text.\n\n## 4. Affected Components\n\nA table.\n"
    )
    section = _markdown_section(content, "## 3. Target Architecture", "## 4. Affected Components")
    assert section == "## 3. Target Architecture\n\nSome diagram text."


def test_markdown_section_returns_empty_when_the_start_heading_is_missing():
    content = "# ADR\n\nNo matching heading here."
    assert _markdown_section(content, "## 3. Target Architecture", "## 4. Affected Components") == ""
