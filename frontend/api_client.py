"""Thin HTTP wrapper around the backend API in `app/routers/frontend.py`.
All UI calls go through these functions. This keeps the API shape in one place.

`BACKEND_URL` points to the FastAPI service. See GETTING_STARTED.md, "Frontend usage".
"""

import os

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8010")


def login(username: str, password: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/login", json={"username": username, "password": password}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def get_config() -> dict:
    response = requests.get(f"{BACKEND_URL}/v1/frontend/config", timeout=10)
    response.raise_for_status()
    return response.json()


def upload_transcription(
    ingestion_date: str,
    files: list[tuple[str, bytes]],
    max_questions_per_stage: int,
    username: str,
) -> dict:
    """No `tenant` field. The backend derives it from `username`
    (`app.routers.frontend._tenant_for_username`). This stops a request from picking a
    different tenant."""
    multipart_files = [("files", (name, content)) for name, content in files]
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/upload",
        data={
            "ingestion_date": ingestion_date,
            "max_questions_per_stage": max_questions_per_stage,
            "username": username,
        },
        files=multipart_files,
        timeout=600,  # A graph run makes many LLM calls in sequence.
    )
    response.raise_for_status()
    return response.json()


def resume_transcription(thread_id: str, answers: dict[str, str], username: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/resume",
        json={"thread_id": thread_id, "answers": answers, "username": username},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def regenerate_document(source_component: str, feedback: str, username: str, current_document: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/regenerate",
        json={
            "source_component": source_component,
            "feedback": feedback,
            "username": username,
            "current_document": current_document,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def ask_more_questions(
    username: str, source_component: str, max_questions_per_stage: int, current_document: str, feedback: str = ""
) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/ask-more",
        json={
            "username": username,
            "source_component": source_component,
            "max_questions_per_stage": max_questions_per_stage,
            "current_document": current_document,
            "feedback": feedback,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def finalize_document(username: str, source_component: str, content: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/finalize",
        json={"username": username, "source_component": source_component, "content": content},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def architecture_history(username: str) -> dict:
    response = requests.get(
        f"{BACKEND_URL}/v1/frontend/architecture-history", params={"username": username}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def chat(username: str, question: str, k: int = 8, history: list[tuple[str, str]] | None = None) -> dict:
    """`history` holds prior turns as `(role, content)` pairs, oldest first. `role` is
    `"user"` or `"assistant"`. Exclude the current `question`. This lets the backend resolve
    follow-up references, for example "who approved it?"."""
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/chat",
        json={
            "username": username,
            "question": question,
            "k": k,
            "history": [{"role": role, "content": content} for role, content in (history or [])],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def test_monitor(username: str) -> dict:
    response = requests.get(
        f"{BACKEND_URL}/v1/frontend/test-monitor", params={"username": username}, timeout=10
    )
    response.raise_for_status()
    return response.json()
