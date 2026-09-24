# Agents / endpoints — LLM call flow

Every place this codebase actually calls an LLM today, and the exact prompt each call uses.
Complements `doc/silver_process.md` (which covers the Silver graph alone, node by node) with the
wider picture: HTTP endpoints, CLI entry points, and where each one does or doesn't touch an LLM.

```mermaid
flowchart TD
    subgraph EP["HTTP endpoints — app/main.py"]
        H1["GET /health<br/>no LLM"]
        H3["POST /v1/completions<br/>llm.router.complete(messages)<br/>LLM call · caller-supplied prompt, no fixed file"]
    end

    subgraph FE["POST /v1/frontend/* — app/routers/frontend.py (backs the Streamlit app)"]
        F1["POST /login, GET /config<br/>no LLM"]
        F2["POST /transcriptions/upload<br/>runs the full Silver graph, persist=False"]
        F3["POST /transcriptions/resume<br/>resumes the paused graph (Command(resume=answers))"]
        F4["POST /transcriptions/regenerate<br/>standalone Actor+Critic call, no Boss, no persistence"]
        F5["POST /transcriptions/ask-more<br/>standalone questions+classification call, no persistence"]
        F6["POST /transcriptions/finalize · Publish<br/>no LLM itself · triggers gold.extract_and_persist_gold_facts"]
        F7["GET /architecture-history<br/>no LLM · gold.current_architecture_diagram (built, not LLM-drawn)"]
        F8["POST /chat<br/>RAG over Gold — see CHAT subgraph"]
        F9["GET /test-monitor<br/>no LLM"]
    end

    subgraph CLI["CLI entry points — Makefile"]
        M2["make questions[-arch|-data-contracts]<br/>scripts/questions.py"]
        M3["make clarify<br/>scripts/clarify.py"]
        M4["make chat<br/>scripts/chat_gold.py · REPL, same retrieval/answer code as /chat"]
    end

    F2 -. same graph as .-> M3

    subgraph GRAPH["agents/graph.py — Silver LangGraph"]
        direction TB
        N1["load_bronze<br/>no LLM"]
        N2["generate_architecture_questions<br/>LLM call #1"]
        N2b["generate_data_contract_questions<br/>LLM call #2 · skipped, zero cost, if node 2\nidentified no contracts"]
        N3["classify_questions<br/>LLM call #3"]
        D1{"pending_questions?"}
        N4["ask_human<br/>interrupt() · no LLM"]
        N5["synthesize_document<br/>LLM call #4 · per source<br/>shallow retry: placeholder leak (blind resample)<br/>or dropped [SUGGEST INFO] (named-correction retry)"]
        N6["critic_document<br/>LLM call #5 · per source, always runs"]
        N6b["boss_decide<br/>no LLM · deterministic policy over Critic claims"]
        D2{"material<br/>contradiction,<br/>1st pass?"}
        N7["ask_human<br/>interrupt() · no LLM"]
        N8["write_document<br/>no LLM · hash-versioned upsert"]
        D3{"persist?"}
        N9["chunk_and_embed<br/>no LLM · local embed(), 1 chunk per ADR"]
        N10["extract_gold_facts<br/>LLM call #6 · per source · skipped if already extracted"]
        N11["resolve_gold_identity<br/>no LLM · exact alias match, then pg_trgm fuzzy match"]
        N12["persist_gold_evolution<br/>no LLM · hash-compare, version bump per entity"]

        N1 --> N2 --> N2b --> N3 --> D1
        D1 -->|yes| N4 --> N5
        D1 -->|no| N5
        N5 --> N6 --> N6b --> D2
        D2 -->|yes| N7 --> N5
        D2 -->|no, or already retried| N8 --> D3
        D3 -->|no · frontend draft| END1(["END · draft only,\nnothing in Silver/Gold\nbeyond the audit trail"])
        D3 -->|yes| N9 --> N10 --> N11 --> N12 --> END2(["END"])
    end

    M2 -. calls both generate_*_questions_for_batch directly .-> N2
    M3 --> N1
    F2 -. runs the graph above .-> N1
    F3 -. resumes the graph above .-> N4

    N2 -. prompt .-> P1["prompts/architecture_questions/combined.jinja<br/>+ prompts/architecture_questions/template.md<br/>(embedded as ARCHITECTURE_CHANGES)<br/>temperature=0, reasoning_effort=none<br/>retries at temp=0.7 on shallow/ungrounded result"]
    N2b -. prompt .-> P1b["prompts/data_contract_questions/questions.jinja<br/>+ prompts/data_contract_questions/template.md<br/>(embedded as DATA_CONTRACT_REQUIREMENTS)<br/>fed node 2's mentioned_data_contracts as IDENTIFIED_DATA_CONTRACTS<br/>same temperature/retry discipline as node 2"]
    N3 -. prompt .-> P2["prompts/classification/classifier.jinja<br/>agents/stages/classification/prompts.py :: build_classification_prompt"]
    N5 -. prompt .-> P3["prompts/adr_generation/generator.jinja<br/>retry turn on dropped [SUGGEST INFO]: SUGGEST_INFO_CORRECTION_MESSAGE<br/>agents/stages/adr_generation/service.py:137-145 — INLINE, no .jinja/.md file"]
    N6 -. prompt .-> P4["prompts/adr_critic/critic.jinja<br/>agents/stages/adr_critic/prompts.py :: build_critic_prompt"]
    N10 -. prompt .-> P5["prompts/gold/extraction.jinja<br/>agents/stages/gold/service.py :: extract_gold_facts_for_source"]

    F4 -. prompt .-> P3
    F4 -. prompt .-> P4
    F5 -. prompt .-> P1
    F5 -. prompt .-> P1b
    F5 -. prompt .-> P2
    F6 -. prompt .-> P5b["gold.extract_and_persist_gold_facts<br/>same extraction.jinja as N10 — the standalone caller for a\ndraft ADR that never went through the graph's own Gold nodes"]

    subgraph CHAT["POST /chat · make chat (scripts/chat_gold.py) — RAG over Gold"]
        direction TB
        C0["_contextualize_question<br/>no LLM · folds recent history into the retrieval text only"]
        C1{"is_evolution_question(text)<br/>AND find_entity_by_name_in_text matches?<br/>no LLM · keyword heuristic + alias lookup"}
        C2["entity_history<br/>no LLM · full version history for one entity"]
        C3["answer_evolution_question<br/>LLM call · narrates the timeline"]
        C4["embed_question<br/>embedding, not a chat completion"]
        C5["top_k_gold_evolution(mode='hybrid')<br/>no LLM:<br/>vector search + lexical ts_rank search<br/>-> Reciprocal Rank Fusion<br/>-> dedupe by entity (keep latest version)<br/>-> [optional] rerank: LOCAL cross-encoder<br/>(ingestion.reranker.score_candidates), NOT an LLM call<br/>-- /chat never passes rerank=True; only 'make chat --rerank' does"]
        C6["answer_question<br/>LLM call · answers strictly from retrieved facts"]

        C0 --> C1
        C1 -->|yes| C2 --> C3
        C1 -->|no| C4 --> C5 --> C6
    end

    F8 --> C0
    M4 --> C0
    C3 -. prompt .-> P6["agents/stages/gold/service.py :: answer_evolution_question<br/>lines ~1187-1206 — INLINE, no .jinja/.md file"]
    C6 -. prompt .-> P7["agents/stages/gold/service.py :: answer_question<br/>lines ~1010-1028 — INLINE, no .jinja/.md file"]

    subgraph EVAL["prompts/architecture_questions/ — decomposed pipeline, evaluation prototype only<br/>(not wired into agents/graph.py or any HTTP endpoint)"]
        E1["identification.jinja → drafting.jinja → selection.jinja<br/>agents/stages/architecture_questions/prompts.py"]
    end
```
