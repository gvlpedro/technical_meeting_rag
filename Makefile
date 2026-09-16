PORT ?= 8010
DATE ?=

INTERACTIVE ?=

# Every test in this repo runs against this database, never against the dev
# database above. This keeps a test run from ever touching real dev data.
TEST_DB_NAME ?= technical_meeting_rag_test
TEST_DATABASE_URL := postgresql+asyncpg://postgres:postgres@localhost:5433/$(TEST_DB_NAME)

.PHONY: test test-acb up down db migrate clean test-db migrate-test ingestion questions questions-arch questions-data-contracts clarify test-gold-arch-evolution chat frontend

db:
	docker compose up -d postgres < /dev/null
	@until docker compose exec -T postgres pg_isready -U postgres < /dev/null >/dev/null 2>&1; do sleep 1; done

migrate: db
	uv run alembic upgrade head

clean: migrate
	PYTHONPATH=. uv run python scripts/clean_dev_data.py

# Creates the test database on first run only. createdb fails if the database
# already exists, so this checks first and skips createdb when it does.
test-db: db
	@docker compose exec -T postgres psql -U postgres -tAc \
		"SELECT 1 FROM pg_database WHERE datname = '$(TEST_DB_NAME)'" < /dev/null | grep -q 1 || \
		docker compose exec -T postgres createdb -U postgres $(TEST_DB_NAME) < /dev/null

migrate-test: test-db
	DATABASE_URL=$(TEST_DATABASE_URL) uv run alembic upgrade head

test: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) uv run pytest

test-arch-questions-acb: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest testing_arch_questions_acb/ -v

test-data-contract-questions-acb: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest testing_data_contract_questions_acb/ -v

test-adr-acb: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest testing_adr_acb/ -v

test-gold-arch-evolution: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest testing_gold_arch_evolution/ -v -s

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

chat: migrate
	PYTHONPATH=. uv run python scripts/chat_gold.py

frontend:
	@echo "Backend must be running separately — 'make up' (docker) or 'uv run uvicorn app.main:app --reload'."
	STREAMLIT_SERVER_HEADLESS=true uv run streamlit run frontend/app.py

