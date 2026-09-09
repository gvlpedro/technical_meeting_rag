PORT ?= 8010
DATE ?=

INTERACTIVE ?=

.PHONY: test test-acb up down db migrate ingestion questions questions-arch questions-data-contracts clarify

db:
	docker compose up -d postgres < /dev/null
	@until docker compose exec -T postgres pg_isready -U postgres < /dev/null >/dev/null 2>&1; do sleep 1; done

migrate: db
	uv run alembic upgrade head

test: migrate
	uv run pytest

test-arch-questions-acb:
	PYTHONPATH=. uv run pytest testing_arch_questions_acb/ -v

test-data-contract-questions-acb:
	PYTHONPATH=. uv run pytest testing_data_contract_questions_acb/ -v

test-adr-acb:
	PYTHONPATH=. uv run pytest testing_adr_acb/ -v

down:
	docker compose down

up: down
	HOST_PORT=$(PORT) docker compose up -d --build
	@echo "Server up, verify on http://localhost:$(PORT)/health"

ingestion: migrate
	@test -n "$(DATE)" || (echo "Usage: make ingestion DATE=YYYYMMDD" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/ingest.py --ingestion-date $(DATE)

questions: migrate
	@test -n "$(DATE)" || (echo "Usage: make questions DATE=YYYYMMDD" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/questions.py --ingestion-date $(DATE)

questions-arch: migrate
	@test -n "$(DATE)" || (echo "Usage: make questions-arch DATE=YYYYMMDD" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/questions.py --ingestion-date $(DATE) --stage architecture

questions-data-contracts: migrate
	@test -n "$(DATE)" || (echo "Usage: make questions-data-contracts DATE=YYYYMMDD" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/questions.py --ingestion-date $(DATE) --stage data-contract

clarify: migrate
	@test -n "$(DATE)" || (echo "Usage: make clarify DATE=YYYYMMDD [INTERACTIVE=1]" >&2; exit 1)
	PYTHONPATH=. uv run python scripts/clarify.py --ingestion-date $(DATE) $(if $(INTERACTIVE),--term,)
