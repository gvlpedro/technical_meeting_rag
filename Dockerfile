FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY llm ./llm
COPY db ./db
COPY ingestion ./ingestion
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY tests ./tests
# app/routers/frontend.py runs the real Silver+Gold graph, so the app image needs the same
# two packages the CLI scripts always needed: agents/ (the graph itself) and prompts/ (the
# Jinja templates agents/service.py and agents/graph.py load from disk at runtime).
COPY agents ./agents
COPY prompts ./prompts
# One image serves both `docker-compose.yml` services (app and frontend) — the Streamlit UI
# is just a different CMD (see that file's `frontend` service) over the same dependency set,
# since streamlit/requests are already ordinary (non-dev) pyproject.toml dependencies.
COPY frontend ./frontend
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
