PORT ?= 8000
PID_FILE := .uvicorn.pid
LOG_FILE := .uvicorn.log

.PHONY: test up down

test:
	uv run pytest

down:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if ps -p $$PID -o command= 2>/dev/null | grep -q "uvicorn app.main:app"; then \
			kill $$PID && echo "Stopped previous server (PID $$PID)"; \
		fi; \
		rm -f $(PID_FILE); \
	fi

up: down
	@nohup uv run uvicorn app.main:app --host 0.0.0.0 --port $(PORT) > $(LOG_FILE) 2>&1 < /dev/null & echo $$! > $(PID_FILE)
	@echo "Server up, verify on http://localhost:$(PORT)/health (PID $$(cat $(PID_FILE))), logs: $(LOG_FILE)"
