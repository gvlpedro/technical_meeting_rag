"""This file makes every test run against the test database. It never runs tests
against the dev database.

The dev database and the test database live on the same Postgres server. But
they are two separate databases: technical_meeting_rag and
technical_meeting_rag_test. See docker-compose.yml and the Makefile's
`test-db`/`migrate-test` targets.

pytest loads this file before it loads any test file. This file sets
DATABASE_URL here. So every test file sees the test database as soon as it
reads app.config.settings.

The Makefile already sets DATABASE_URL explicitly for every test target. This
file is a safety net for a test run that skips the Makefile. For example, a
test run started from an IDE or a bare `pytest` command. os.environ.setdefault
only sets the value when it is not already set. So this file never overrides
the Makefile's own value.
"""

import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5433/technical_meeting_rag_test",
)
