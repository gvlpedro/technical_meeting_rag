# technical_meeting_rag

Technical meetings about a system's architecture are scattered across many separate conversations, each told from a different angle — a backend engineer describing a service, a data engineer describing a pipeline, a frontend engineer describing a view — and each capturing only a partial, informal snapshot of the truth at that moment.

This project builds a **Medallion RAG Architecture** that ingests transcripts and a more clarified version as queryable knowledge base.

# Objective

For every architecture component mentioned across meetings, clarify **what it is**, whether it is **new**, an **evolution** of something already known, or **unchanged**, and produce two things Gold makes: an **Architecture Decision Record (ADR)** per evolution — capturing the context that motivated it, the alternatives considered, the trade-offs accepted, and the decision itself — and a **versioned record of each component**, instead of a flat, undated summary of what was said.

# Problems to resolve

* **Fragmented knowledge:** the same architecture gets described across dozens of separate meetings, with no single place that reflects the current, agreed-upon picture.
* **Ambiguity:** statements made in a meeting are often underspecified and cannot be trusted as a final definition of a component without further clarification.
* **Contradictions:** different meetings — or different people in the same meeting — describe the same component inconsistently, and nothing flags the conflict.
* **Architecture evolution over time:** components are described at different points in time; without ordering that timeline, it is unclear which description is still valid.
* **Implemented vs. planned confusion:** meetings mix what already exists with what is only intended, and that distinction is easy to lose once everything is summarized together.
* **Undocumented boundaries between teams:** the contracts and dependencies between profiles (e.g. what Data Engineering expects from Software Engineering) are usually implicit, never written down anywhere.
* **Restricted visibility:** some information is only meant to be seen by certain profiles due to organizational or permission boundaries, and a single flattened summary would leak or ignore that distinction.
* **Lack of traceability:** once a meeting is summarized by hand, it is normally impossible to trace a statement back to who said it, when, and in which conversation.
* **Manual alignment does not scale:** reconciling all of the above by hand, meeting after meeting, does not scale as the organization and its architecture grow.

# Clarification process: Agent & Human-in-the-loop collaboration

The project refine the final understanding of the organization asking to clarify following points:

1. **Clarify ambiguities:** Ask for items that are not clear enough and require additional information to properly define each component.
2. **Clarify contradictions:** Ask for items that are inconsistent between different tellings of the same component and require clarification or resolution.
3. **Clarify architecture evolution:** Clarify the timeline, components are described in different moments so project must order the evolution.
4. **Clarify subsystems:** Some components are described as part of a bigger system, so project must ask the boundaries and dependencies between them.
5. **Clarify implemented vs. planned:** Ask if components and capabilities that already exist and those that have not been implemented yet.

The goal is not simply to summarize a meeting, but to produce a **consistent, clarified record of the organization's architecture**, while explicitly identifying gaps, boundaries, dependencies, and inconsistencies.

Check document [doc/silver_process.md](doc/silver_process.md) for more details.

## Workflow

Input transcription > Clarification (LLM / Human) > Pull Request > Enrich RAG 

Although the workflow shows how user understands the flow, internally data follows a layered approach in PostgreSQL due to raw transcriptions are processed to store the raw data and then we clarify the information in a silver layer and finally we cover a synthesis of the architecture in gold layer.

```
                    ┌──────────────────────┐
                    │      RAW / Bronze    │
                    │                      │
Documents ─────────►│ original content     │
                    │ metadata             │
                    └──────────┬───────────┘
                               │
                               │ clarification agent process
                               ▼
                    ┌───────────────────────────┐
                    │ SILVER / Clarified input  │
                    │                           │
                    │ Summary chunking          │
                    │ metadata enritchment      │
                    │ generated data contracts  │
                    └──────────┬────────────────┘
                               │
                               │ semantic refinement
                               ▼
                    ┌────────────────────────────────────┐
                    │ GOLD / ADR: versioned components   │
                    │                                    │
                    │ Structure-aware chunking           │
                    │ Pull request / versions            │
                    │ metadata enritchment               │
                    │ Component graph                    │
                    └────────────────────────────────────┘
```

## Expected questions to resolve

* When the component X was introduced in the company?
* Show me the general architecture diagram
* Who is responsible for the component X?
* Add a new component Y with [X, Z] dependencies and following model [...]
* Let me know the list of persons talking about the component X
* Let me know the details of the component X, including its dependencies and boundaries

## Guardrails

Identify items that are hidden due to permissions or organizational boundaries, where some components are visible to certain profiles but not to others.

## Samples

### Synthetic transcriptions
Specific descriptions for managing unit tests and verify expected behavior from different descriptions.

### Youtube
Only to process real transcriptions that are very different from each other the youtube transcriptions will be used as input.
These transcripts will be processed to generate the clarified architecture record, applying the same guardrails to identify missing information, inconsistencies, implementation status, and boundaries between profiles.

```
# Download a video's transcript; its session is derived from the video's own
# YouTube upload_date (input/transcriptions/session=<upload_date>/)
python3 scripts/download_transcript.py "https://www.youtube.com/watch?v=04uC4zrU10k"

# Re-download every link already tracked in input/links.json
python3 scripts/download_transcript.py

# ...or only the ones uploaded on a given date
python3 scripts/download_transcript.py --session 20260906
```

## Output

First of all the project creates a 'Pull request' based on unclarified components, missing information and inconsistencies.

Then this will generate:
* One general architecture diagram of the components discussed
* A complete documentation clasified per component and timeline

## User interface

Once user authenticates in a session (isolating information) to show different tabs:

* Input transcription
* Test  monitor
* Pull request to clarify new information
* Chat with RAG 

## Main stack

* Python 3.11
* Docker
* LangChain
* PgVector
* Streamlit
* mermaid-cli
* datacontract-cli

### Libraries

    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "openai>=1.0",
    "anthropic>=0.40",
    "python-dotenv>=1.0",
    "structlog>=24.0",
    "litellm>=1.50",
    "sse-starlette>=2.1",
    "jinja2>=3.1",
    "instructor>=1.6",
    "python-multipart>=0.0.9",
    "pypdf>=4.0",
    "reportlab>=4.0",
    "tiktoken>=0.7.0",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.29",
    "pgvector>=0.3",
    "alembic>=1.13",
    "sentence-transformers>=3.0",
    "ragas>=0.2",
    "mermaid-py>=0.8.4",
    "open-data-contract-standard>=3.0.1"

# Architecture

A single FastAPI application backed by Postgres (`pgvector` for embeddings) and a LangGraph agent
for the human-in-the-loop clarification step; Streamlit is the planned UI layer (see `## User
interface` above — not yet wired, tracked in `.tmp/tasks.md`).

Storage follows the Medallion layering shown in the diagram under `## Workflow` above, one Postgres
table set per layer:

| Layer  | Table(s)                            |
| ------ |-------------------------------------|
| Bronze | `bronze_documents`                  |
| Silver | `silver_documents`, `silver_chunks` | 
| Gold   | `adr`,`components`                  | 

# Key Decisions

Engineering decisions made for this project itself — not to be confused with the ADRs the pipeline
produces *about the meetings it ingests* (that's the Objective above):

* **Medallion layering (Bronze → Silver → Gold), not one flat store.** Each layer has a narrower,
  independently-verifiable job — Bronze never interprets, Silver clarifies, Gold records the ADRs
  (`adr`) and each component's version history (`components`)
* **Clarification questions are drafted per transcript, not read from a fixed list.** One LLM call
  reads the transcript against `doc/clarification_template.md` and asks one specific question per
  component actually named ("Is `X` new, evolving, or unchanged?"), instead of one generic question
  that silently covers every component at once (`prompting/roles/common/clarification_questions.jinja`).
* **Clarification is LLM-first, human-in-the-loop only when needed.** One classification call
  sorts every drafted question into `answered` / `unknown` / `needs_clarification`; only the last
  group reaches a human, batched into a single LangGraph `interrupt()`
* **LiteLLM router with OpenAI → Anthropic fallback**, so a single provider outage doesn't stop
  ingestion or clarification.

# Key Metrics

Quality gates defined for the RAG pipeline (Phase 6 of `.tmp/tasks.md` — not yet implemented;
tracked here so the target is explicit before the tests are written):

| Metric                                       | What it catches                                                                    | Threshold |
| --------------------------------------------- | ----------------------------------------------------------------------------------- | --------- |
| Top-k / distance-metric / filter correctness  | Off-by-one top-k, wrong similarity ordering, a profile/session filter letting the wrong chunks through | Exact match against hand-computed expectations |
| ANN index recall@k vs. brute-force            | An under-tuned pgvector index (HNSW/IVFFlat) silently dropping the one chunk that mattered | ≥ 0.95 |
| RAGAS faithfulness                            | The answer contains a claim the retrieved chunks don't support (hallucination)     | Documented per-metric minimum, gates merges (`eval/thresholds.yaml`) |
| RAGAS context precision                       | Retrieved chunks are mostly irrelevant padding                                     | ″ |
| RAGAS context recall                          | Retrieved chunks miss something the reference answer needed                        | ″ |
| Guardrail leakage                             | A profile-restricted chunk reaches an answer generated for a different profile, even paraphrased | Zero tolerance — any leak is a fail, not a threshold |


# Integration

MCP for each output for internal agents

