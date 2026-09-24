"""Fast, deterministic tests for `agents/shared.py`. This file holds the cross-stage helpers
that more than one pipeline stage depends on. This test file stays under `tests/`, not under
any `agents/stages/<stage>/test/` folder, because no single stage owns this code."""

from agents.shared import (
    extract_source_line,
    insert_authors_line,
    insert_source_line,
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
