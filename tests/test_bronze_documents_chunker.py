from datetime import date

import pytest

from ingestion.bronze_documents_chunker import parse_ingestion_date


def test_parse_ingestion_date_parses_compact_yyyymmdd():
    assert parse_ingestion_date("20260906") == date(2026, 9, 6)


def test_parse_ingestion_date_raises_for_non_date_input():
    with pytest.raises(ValueError):
        parse_ingestion_date("not-a-date")


def test_parse_ingestion_date_raises_for_invalid_date_digits():
    with pytest.raises(ValueError):
        parse_ingestion_date("99999999")
