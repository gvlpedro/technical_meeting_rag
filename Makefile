PORT ?= 8010
DATE ?=
# Not named TERM: that's already a standard shell env var (terminal type, e.g.
# xterm-256color) present in every interactive session — `TERM ?=` would never
# override it, so `$(if $(TERM),...)` would always be true. INTERACTIVE avoids the
# collision.
INTERACTIVE ?=

.PHONY: test test-acb up down db migrate ingestion questions clarify

db:
	docker compose up -d postgres < /dev/null
	@until docker compose exec -T postgres pg_isready -U postgres < /dev/null >/dev/null 2>&1; do sleep 1; done

migrate: db
	uv run alembic upgrade head

test: migrate
	uv run pytest

# Real-LLM golden-set question collection (see testing_questions_acb/README.md)pwd
test-questions-acb:
	PYTHONPATH=. uv run pytest testing_questions_acb/ -v

# Real-LLM golden-set ADR generation (see testing_adr_acb/README.md)
test-adr-acb:
	PYTHONPATH=. uv run pytest testing_adr_acb/ -v

# Real-LLM single-case test (see testing_questions_for_one_component/README.md)
test-minimal-new-component:
	PYTHONPATH=. uv run pytest testing_questions_for_one_component/ -v

down:
	docker compose down

up: down
	HOST_PORT=$(PORT) docker compose up -d --build
	@echo "Server up, verify on http://localhost:$(PORT)/health"

# make ingestion DATE=20260906
ingestion: migrate
	@test -n "$(DATE)" || (echo "Usage: make ingestion DATE=YYYYMMDD" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/ingest.py --ingestion-date $(DATE)

# make questions DATE=20260906
questions: migrate
	@test -n "$(DATE)" || (echo "Usage: make questions DATE=YYYYMMDD" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/questions.py --ingestion-date $(DATE)

# make clarify DATE=20260906 [INTERACTIVE=1]
# INTERACTIVE=1 answers clarification questions interactively in this terminal (passes
# scripts/clarify.py's own --term flag); without it, an interrupt just prints the
# pending questions and exits (nothing to resume with yet).
clarify: migrate
	@test -n "$(DATE)" || (echo "Usage: make clarify DATE=YYYYMMDD [INTERACTIVE=1]" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/clarify.py --ingestion-date $(DATE) $(if $(INTERACTIVE),--term,)
