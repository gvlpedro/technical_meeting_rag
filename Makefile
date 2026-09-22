PORT ?= 8010
FRONTEND_PORT ?= 8522
DATE ?=

INTERACTIVE ?=

# Every test in this repo runs against this database, never against the dev
# database above. This keeps a test run from ever touching real dev data.
TEST_DB_NAME ?= technical_meeting_rag_test
TEST_DATABASE_URL := postgresql+asyncpg://postgres:postgres@localhost:5433/$(TEST_DB_NAME)

# The dev database name, matching docker-compose.yml's own `POSTGRES_DB`. `backup`/`restore`
# always target this one, never $(TEST_DB_NAME) — a restore is destructive (`pg_restore
# --clean`), and a test run already gets a throwaway database via `migrate-test` on every run,
# so it never needs restoring from a file to begin with.
DEV_DB_NAME ?= technical_meeting_rag
BACKUP_DIR ?= backups
FILE ?=

.PHONY: test test-acb up down db migrate clean test-db migrate-test ingestion questions questions-arch questions-data-contracts clarify test-gold-arch-evolution chat frontend backup restore

db:
	docker compose up -d postgres < /dev/null
	@until docker compose exec -T postgres pg_isready -U postgres < /dev/null >/dev/null 2>&1; do sleep 1; done

migrate: db
	uv run alembic upgrade head

clean: migrate
	PYTHONPATH=. uv run python scripts/clean_dev_data.py

# `-Fc` (custom format), not plain SQL: it lets `pg_restore` below run `--clean --if-exists`
# safely against a database that still has its own schema in place, and it dumps every
# extension this database actually uses (vector, pg_trgm) as `CREATE EXTENSION IF NOT EXISTS`,
# so a restore never fails on an extension that happens to already be installed.
#
# `FILE` is optional here, unlike `restore`. Given, it overrides the default timestamped path
# under $(BACKUP_DIR); left unset, `make backup` always works with no arguments.
backup: db
	@mkdir -p $(BACKUP_DIR)
	@out="$(FILE)"; \
	if [ -z "$$out" ]; then out="$(BACKUP_DIR)/$(DEV_DB_NAME)_$$(date +%Y%m%d_%H%M%S).dump"; fi; \
	docker compose exec -T postgres pg_dump -U postgres -d $(DEV_DB_NAME) -Fc < /dev/null > "$$out"; \
	echo "Backup written to $$out"

# Destructive: `--clean --if-exists` drops every existing table, index, and row in
# $(DEV_DB_NAME) before restoring this file's own copy of each. This never touches
# $(TEST_DB_NAME) — see that variable's own comment above. `--no-owner --no-privileges` drops
# the dump's own OWNER TO / GRANT statements, so a restore never fails just because it is
# replayed by a different role than the one that created the backup.
restore:
	@test -n "$(FILE)" || (echo "Usage: make restore FILE=path/to/backup.dump" >&2; exit 1)
	@test -f "$(FILE)" || (echo "No such file: $(FILE)" >&2; exit 1)
	$(MAKE) db
	@echo "Restoring $(FILE) into $(DEV_DB_NAME) — this REPLACES all current data in that database."
	docker compose exec -T postgres pg_restore -U postgres -d $(DEV_DB_NAME) --clean --if-exists --no-owner --no-privileges < $(FILE)
	@echo "Restored from $(FILE)"

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

test-classifier-acb: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest agents/stages/classification/testing/ -v

test-arch-questions-acb: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest agents/stages/architecture_questions/testing/ -v

test-data-contract-questions-acb: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest agents/stages/data_contract_questions/testing/ -v

test-adr-acb: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest agents/stages/adr_generation/testing/ -v

test-gold-arch-evolution: migrate-test
	DATABASE_URL=$(TEST_DATABASE_URL) PYTHONPATH=. uv run pytest agents/stages/gold/testing/ -v -s

down:
	docker compose down

up: down
	HOST_PORT=$(PORT) FRONTEND_HOST_PORT=$(FRONTEND_PORT) docker compose up -d --build
	@echo "Backend up, verify on http://localhost:$(PORT)/health"
	@echo "Frontend starting (waits on the backend's own healthcheck first) — give it a few"
	@echo "seconds, then open http://localhost:$(FRONTEND_PORT)"

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
	@echo "For local development only, with live reload on save — 'make up' alone already runs"
	@echo "the frontend in Docker too, at http://localhost:$(FRONTEND_PORT)."
	@echo "Backend must be running separately — 'make up' (docker) or 'uv run uvicorn app.main:app --reload'."
	STREAMLIT_SERVER_HEADLESS=true uv run streamlit run frontend/app.py

