# Getting Started

## Start project

```bash
make up
```

## Tests

In total there are 13 tests available

```bash
make test
```

## Useful commands


Ingest youtube transcriptions
```bash
make ingestion DATE=20260515
```

Generate architecture questions for sessions on date 2024-05-15 in Silver (interactive mode) :
```bash
make questions-arch DATE=20260515
```

Generate data contract questions for sessions on date 2024-05-15 in Silver (interactive mode) :
```bash
make questions-data-contracts DATE=20260515
```

Complete classification process (question generation / human answers / ADR generarions) for sessions on date 2024-05-15 in Silver :
```bash
make clarify DATE=20260515 INTERACTIVE=1
```

Chat with the agent (interactive mode) :
```bash
make chat
```

## ACTOR CRITIC BOSS TESTS

Make sure the schema is up to date first:
```bash
make migrate
```

Testing actor-critic-boss for architecture questions
```bash
make test-adr-acb
```

Testing actor-critic-boss for data contract questions
```bash
make test-data-contract-acb
```

Testing actor-critic-boss for data contract questions
```bash
make test-data-contract-acb
```

Automated E2E test — ingests 5 sequential real transcripts (~10 min)
```bash
make test-gold-arch-evolution
```

## Frontend usage

The Streamlit frontend (`frontend/`) is the "User interface" described in the root
README — one tab each for uploading a transcript, reviewing/accepting its ADR, chatting
with Gold, and (only when `start_test_mode` is on) monitoring test results and LLM cost.

There is no real user system — `app/config.py`'s `frontend_users` is a fixed, hardcoded
roster. Each login maps to its own **tenant**, which is what actually isolates data: a file
`pepe` uploads, or an ADR `pepe` accepts, is invisible to `peter`, and vice versa.

| Username | Password | Tenant  |
| -------- | -------- | ------- |
| `pepe`   | `1234`   | `lidr`  |
| `peter`  | `123`    | `lotus` |

Run the backend first, then the frontend in a second terminal:

```bash
make up          # backend on http://localhost:8010 (docker), or:
# uv run uvicorn app.main:app --reload   # backend without docker

make frontend     # Streamlit on http://localhost:8501
```

If the backend runs somewhere other than `http://localhost:8010`, point the frontend at it
with `BACKEND_URL`:

```bash
BACKEND_URL=http://localhost:8010 make frontend
```

Log in as `pepe`/`1234` or `peter`/`123`, then:

1. **Input transcription** — upload a `.txt`, `.vtt`, `.md`, or `.pdf` for one meeting and
   click "Process and clarify". If the clarification loop needs a human answer, the tab shows the pending
   questions right there — answer them and submit to finish the run. Once you're happy with the
   generated ADR, click "Publish" — its facts are in Gold and retrievable from Chat immediately.
2. **Architecture history** — every ADR published so far, in a table (click "View ADR" to open
   one in a new tab), plus a live Mermaid diagram of the current architecture built straight
   from Gold's own components — each node names the ADR that last touched it.
3. **Chat with RAG** — ask about the architecture. Retrieves from every fact published so far,
   scoped to your own tenant.
4. **Test monitor** (only visible when `start_test_mode: true` in `app/config.py`, the
   default) — the four `testing_*/output/result.json` files, plus every real LLM call's
   token/cost usage (`output/llm_usage.jsonl`), broken down by tenant.