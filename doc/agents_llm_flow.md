# Agents / endpoints — LLM call flow

Every place this codebase actually calls an LLM today, and the exact prompt each call uses.
Complements `doc/silver_process.md` (which covers the Silver graph alone, node by node) with the
wider picture: HTTP endpoints, CLI entry points, and where each one does or doesn't touch an LLM.

```mermaid
flowchart TD
    subgraph EP["HTTP endpoints — app/main.py"]
        H1["GET /health<br/>no LLM"]
        H2["POST /v1/ingest<br/>ingestion.service.ingest_bronze<br/>no LLM — parse_vtt + chunk_text + local embed()"]
        H3["POST /v1/completions<br/>llm.router.complete(messages)<br/>LLM call · caller-supplied prompt, no fixed file"]
    end

    subgraph CLI["CLI entry points — Makefile"]
        M1["make ingestion<br/>scripts/ingest.py"]
        M2["make questions<br/>scripts/questions.py"]
        M3["make clarify<br/>scripts/clarify.py"]
    end

    M1 -. same service as .-> H2

    subgraph GRAPH["agents/graph.py — Silver LangGraph (make clarify only; no HTTP surface yet)"]
        direction TB
        N1["load_bronze<br/>no LLM"]
        N2["generate_architecture_questions<br/>LLM call #1"]
        N2b["generate_data_contract_questions<br/>LLM call #2 · skipped, zero cost, if node 2\nidentified no contracts"]
        N3["classify_questions<br/>LLM call #3"]
        D1{"pending_questions?"}
        N4["ask_human<br/>interrupt() · no LLM"]
        N5["synthesize_document<br/>LLM call #4 · per source"]
        N6["critic_document<br/>LLM call #5 · per source, always runs"]
        D2{"material<br/>contradiction?"}
        N7["ask_human<br/>interrupt() · no LLM"]
        N8["write_document<br/>no LLM · hash-versioned upsert"]
        N9["chunk_and_embed<br/>no LLM · local embed(), 1 chunk per ADR"]

        N1 --> N2 --> N2b --> N3 --> D1
        D1 -->|yes| N4 --> N5
        D1 -->|no| N5
        N5 --> N6 --> D2
        D2 -->|yes, 1st pass| N7 --> N5
        D2 -->|no, or already retried| N8 --> N9
    end

    M2 -. calls both generate_*_questions_for_batch directly .-> N2
    M3 --> N1

    N2 -. prompt .-> P1["prompts/architecture_questions.jinja<br/>+ prompts/architecture_template.md<br/>(embedded as ARCHITECTURE_CHANGES)<br/>temperature=0, reasoning_effort=none<br/>retries at temp=0.7 on shallow/ungrounded result"]
    N2b -. prompt .-> P1b["prompts/data_contract_questions.jinja<br/>+ prompts/data_contract_template.md<br/>(embedded as DATA_CONTRACT_REQUIREMENTS)<br/>fed node 2's mentioned_data_contracts as IDENTIFIED_DATA_CONTRACTS<br/>same temperature/retry discipline as node 2"]
    N3 -. prompt .-> P2["agents/prompts.py :: build_classification_prompt<br/>(inline system+user strings, no .jinja file)"]
    N5 -. prompt .-> P3["prompts/adr_generator.jinja"]
    N6 -. prompt .-> P4["agents/prompts.py :: build_critic_prompt<br/>(inline system+user strings, no .jinja file)"]

    subgraph UNUSED["prompts/ — not called by any code path today"]
        U1["mermaid_diagram.md<br/>manual/external use only"]
        U2["data_contracts.md<br/>manual/external use only —<br/>scripts/validate_data_contracts.py only lints<br/>its *output*, never renders this prompt"]
    end
```


