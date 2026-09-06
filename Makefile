PORT ?= 8010

.PHONY: test up down

test:
	uv run pytest

down:
	docker compose down

up: down
	HOST_PORT=$(PORT) docker compose up -d --build
	@echo "Server up, verify on http://localhost:$(PORT)/health"
