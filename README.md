# organization_meeting_rag

Technical meetings about a system's architecture are scattered across many separate conversations, each told from a different angle — a backend engineer describing a service, a data engineer describing a pipeline, a frontend engineer describing a view — and each capturing only a partial, informal snapshot of the truth at that moment.

This project builds a Retrieval-Augmented Generation (RAG) system that ingests transcripts and a more clarified version as queryable knowledge base.

# Problems to resolve

* **Fragmented knowledge:** the same architecture gets described across dozens of separate meetings, with no single place that reflects the current, agreed-upon picture.
* **Perspective mismatch:** a generic summary of a meeting is rarely useful on its own — Business, Software Engineering, Data Engineering, and Frontend each need a different, role-specific view of the same information.
* **Ambiguity:** statements made in a meeting are often underspecified and cannot be trusted as a final definition of a component without further clarification.
* **Contradictions:** different meetings — or different people in the same meeting — describe the same component inconsistently, and nothing flags the conflict.
* **Architecture evolution over time:** components are described at different points in time; without ordering that timeline, it is unclear which description is still valid.
* **Implemented vs. planned confusion:** meetings mix what already exists with what is only intended, and that distinction is easy to lose once everything is summarized together.
* **Undocumented boundaries between teams:** the contracts and dependencies between profiles (e.g. what Data Engineering expects from Software Engineering) are usually implicit, never written down anywhere.
* **Restricted visibility:** some information is only meant to be seen by certain profiles due to organizational or permission boundaries, and a single flattened summary would leak or ignore that distinction.
* **Lack of traceability:** once a meeting is summarized by hand, it is normally impossible to trace a statement back to who said it, when, and in which conversation.
* **Manual alignment does not scale:** reconciling all of the above by hand, meeting after meeting, does not scale as the organization and its architecture grow.

# Align different perspectives

First of all, the expected result is an alignment between different profiles in the company, the input are transcriptions of meetings and the alignment will provide a consistent RAG (Retrieval-Augmented Generation) with all distilled information.

This project aims to provide a dedicated perspective for each technical team:

* **Business:** Only business-related details, requirements, and decisions.
* **Software Engineering:** Only components, decisions, and information related to software engineering.
* **Data Engineering:** Only components, decisions, and information related to data engineering and data analytics.
* **Frontend Engineering:** Only components, decisions, and information related to frontend engineering and user experience.
* **Data Contracts:** Common schemas (e.g. OCDS) defining the boundaries and shared items between the different profiles.


# Clarification process: Agent & Human-in-the-loop collaboration

The project refine the final understanding of the organization asking to clarify following points:

1. **Clarify ambiguities:** Ask for items that are not clear enough and require additional information to properly define each perspective.
2. **Clarify contradictions:** Ask for items that are inconsistent between the different perspectives and require clarification or resolution.
3. **Clarify architecture evolution:** Clarify the timeline, components are described in different moments so project must order the evolution.
4. **Clarify subsystems:** Some components are described as part of a bigger system, so project must ask the boundaries and dependencies between them.
2. **Clarify implemented vs. planned:** Ask if components and capabilities that already exist and those that have not been implemented yet.

The goal is not simply to summarize a meeting, but to produce **consistent, role-specific views of the same organization**, while explicitly identifying gaps, boundaries, dependencies, and inconsistencies between those views.

## Workflow

Input transcription > Clarification (LLM / Human) > Pull Request > Enrich RAG 

## Expected questions

* When the component X was introduced in the company?
* Show me the architecture from Software Engineering perspective
* Who is responsible for the component X?
* Let me know the list of persons talking about the component X
* Let me know the details of the component X, including its dependencies and boundaries

## Guardrails

Identify items that are hidden due to permissions or organizational boundaries, where some components are visible to certain profiles but not to others.

## Samples

### Synthetic transcriptions
Specific descriptions for managing unit tests and verify expected behavior from different descriptions.

### Youtube
Only to process real transcriptions that are very different from each other the youtube transcriptions will be used as input.
These transcripts will be processed to generate and compare the different perspectives, applying the same guardrails to identify missing information, inconsistencies, implementation status, and boundaries between profiles.

```
# Add new link to session_1 and download the transcript
python3 scripts/download_transcript.py --session session_1 "https://www.youtube.com/watch?v=04uC4zrU10k"

# Download all links from session_1
python3 scripts/download_transcript.py --session session_1
```

## Output

First of all the project creates a 'Pull request' based on unclarified components, missing information and inconsistencies.

Then this will generate:
* One diagram per perspective
* A complete documentation clasified per component, perspective and timeline

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

# Integration

MCP for each output for internal agents

