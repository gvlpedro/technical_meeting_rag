"""Fast, deterministic tests for `agents/shared.py`. This file holds the cross-stage helpers
that more than one pipeline stage depends on. This test file stays under `tests/`, not under
any `agents/stages/<stage>/test/` folder, because no single stage owns this code."""

from agents.shared import mentions_grounded_in_source


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
