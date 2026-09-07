PORT ?= 8010

.PHONY: test up down db migrate

db:
	docker compose up -d postgres
	@until docker compose exec -T postgres pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done

migrate: db
	uv run alembic upgrade head

test: migrate
	uv run pytest

down:
	docker compose down

up: down
	HOST_PORT=$(PORT) docker compose up -d --build
	@echo "Server up, verify on http://localhost:$(PORT)/health"
