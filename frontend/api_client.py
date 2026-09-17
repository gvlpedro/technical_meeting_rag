"""Thin HTTP wrapper around the backend's `app/routers/frontend.py` endpoints. `app.py` never
builds a request by hand — every call the UI makes goes through one of these functions, so the
shape of the backend API only has to be known in one place.

`BACKEND_URL` points at the FastAPI service (`docker-compose.yml`'s `app`, or a local
`uvicorn app.main:app` run) — see GETTING_STARTED.md's "Frontend usage" section.
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
    tenant: str,
    ingestion_date: str,
    files: list[tuple[str, bytes]],
    max_questions_per_stage: int,
    username: str,
) -> dict:
    multipart_files = [("files", (name, content)) for name, content in files]
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/upload",
        data={
            "tenant": tenant,
            "ingestion_date": ingestion_date,
            "max_questions_per_stage": max_questions_per_stage,
            "username": username,
        },
        files=multipart_files,
        timeout=600,  # a real graph run makes several sequential LLM calls
    )
    response.raise_for_status()
    return response.json()


def resume_transcription(thread_id: str, answers: dict[str, str]) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/resume",
        json={"thread_id": thread_id, "answers": answers},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def regenerate_document(
    tenant: str, source_component: str, feedback: str, username: str, current_document: str
) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/regenerate",
        json={
            "tenant": tenant,
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
    tenant: str, source_component: str, max_questions_per_stage: int, current_document: str, feedback: str = ""
) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/ask-more",
        json={
            "tenant": tenant,
            "source_component": source_component,
            "max_questions_per_stage": max_questions_per_stage,
            "current_document": current_document,
            "feedback": feedback,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def finalize_document(tenant: str, source_component: str, content: str) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/transcriptions/finalize",
        json={"tenant": tenant, "source_component": source_component, "content": content},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def architecture_history(tenant: str) -> dict:
    response = requests.get(
        f"{BACKEND_URL}/v1/frontend/architecture-history", params={"tenant": tenant}, timeout=10
    )
    response.raise_for_status()
    return response.json()


def chat(tenant: str, question: str, k: int = 8) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/v1/frontend/chat",
        json={"tenant": tenant, "question": question, "k": k},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def test_monitor() -> dict:
    response = requests.get(f"{BACKEND_URL}/v1/frontend/test-monitor", timeout=10)
    response.raise_for_status()
    return response.json()
