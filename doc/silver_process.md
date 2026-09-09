# Silver process is a clarification loop to store clarified ADRs in silver layer

Bronze holds raw, unreviewed transcript chunks. Silver's job is to turn one ingestion batch
(`bronze_documents` for a given `ingestion_date`) into **one clarified ADR per source
transcript**, shaped as `prompts/adr_generator.jinja` produces it — filled in as
far as the clarification loop's answers allow, **not necessarily complete** — a document with
several sections collapsed to their "not covered" notes is the expected, correct output when a
transcript doesn't establish a decision, not a defect. Written as **structured Markdown** in
Postgres.

**Idempotency, now versioned:** re-running the process for a source transcript compares the
freshly generated ADR's content hash against the latest stored `silver_documents` row for that
`source_component`. An identical hash overwrites that row in place (same `version`, no
duplicate). A different hash — the transcript's clarifications genuinely changed the resulting
ADR — inserts a new row at `version + 1` and leaves the older version's row and its
`silver_chunks` row alone; both versions stay queryable. `(source_component, version)` is the
real identity now, not `source_component` alone. `silver_clarifications` (below) remains the
deliberate exception to all of this: it's an append-only history, so re-running the process
always adds new rows to it instead of overwriting or versioning.

## 1. Storage

**In Postgres** — the source of truth, three tables. Both question-generation stages (§3 nodes
2-3) also write their own human-readable audit copy of their output, to
`output/ingestion_date=<date>/questions/<transcription>.json` and
`output/ingestion_date=<date>/data_contract_questions/<transcription>.json` respectively —
alongside `write_document`'s own `output/ingestion_date=<date>/adr/<transcription>.md` (§3 node
10), these are on-disk artifacts in Silver, and that's exactly what they are: nothing
downstream reads any of them back, `silver_clarifications`/`silver_documents` (below) stay the
real record.

```
silver_documents                       -- one row per (source transcript, version) clarified ADR
  id                 pk
  ingestion_date     date, not null, index
  source_component   varchar, not null, index    -- e.g. "design_twitter_system_design_interview.en"
                                                   -- — matches bronze_documents.source_component;
                                                   -- no longer unique alone, see version below
  version            integer, not null, default 1 -- 1, 2, 3... per source_component; bumped only
                                                   -- when the generated content_hash actually
                                                   -- changes from the latest stored version
                                                   -- (write_document decides this, never the LLM)
  content_hash       varchar(64), not null        -- sha256 hex of `content` — the deterministic
                                                   -- "is this the same version" check
  content            text, not null              -- prompts/adr_generator.jinja's
                                                 -- structure, filled in as far as the
                                                 -- clarification loop's answers allow (§3 node
                                                 -- 7); uncovered sections collapse to their
                                                 -- prescribed one-sentence notes, never fabricated

  UNIQUE (source_component, version)

silver_clarifications                  -- history of every clarification question and its answer
  id                 pk
  ingestion_date     date, not null, index
  source_component   varchar, not null, index   -- which transcript this question was drafted
                                                 -- against; matches silver_documents.source_component
  question           text, not null
  answer             text, nullable             -- null when the question was never actually
                                                 -- answered (unknown / needs_clarification at write
                                                 -- time)
  answered_at        timestamptz                -- row insertion time, not "when a human typed the
                                                 -- answer" — write_document (§3 node 10) inserts a
                                                 -- fresh row per (source, question) on every run
                                                 -- instead of upserting, so the same question
                                                 -- answered differently across two runs keeps both
                                                 -- answers on record, ordered by this column. Used
                                                 -- to be silver_documents.clarifications (a jsonb
                                                 -- column pooled per ingestion_date); isn't anymore.

silver_chunks                          -- one row per silver_documents (source_component, version)
  id                 pk
  ingestion_date     date, not null, index
  source_component   varchar, not null      -- which silver_documents row this chunk came from; no
                                             -- FK, same flat pattern bronze_documents already uses
  version            integer, not null, default 1  -- matches the silver_documents row's version
  content            text, not null      -- the WHOLE ADR, not a token-split fragment — an ADR is
                                          -- retrieved whole, never split (§3 node 11)
  embedding          vector(settings.embedding_dim)

  UNIQUE (source_component, version)
```


---

## 2. Clarification loop: the decision rule

The question list is about architecture and data contracts in different agent steps to collect questions from the transcript.

| Outcome | Meaning | Action |
|---|---|---|
| `answered` | The transcript states it, or implies it unambiguously. | Record the answer. No human involved. |
| `unknown` | The transcript doesn't say, and it plausibly never will be known from this meeting. | Record `answer = null`. **Do not ask.** |
| `needs_clarification` | The transcript doesn't say, but a human involved would plausibly know. | Batch into the single `interrupt()` call. |

If human doesn't know an answer during resume downgrades that question to `unknown` post hoc — never
re-asked.

The loop should refine the final understanding of the organization asking to clarify following points:

1. **Clarify ambiguities:** Ask for items that are not clear enough and require additional information to properly define each component.
2. **Clarify contradictions:** Ask for items that are inconsistent between different tellings of the same component and require clarification or resolution.
3. **Clarify architecture evolution:** Clarify the timeline, components are described in different moments so project must order the evolution.
4. **Clarify subsystems:** Some components are described as part of a bigger system, so project must ask the boundaries and dependencies between them.
5. **Clarify implemented vs. planned:** Ask if components and capabilities that already exist and those that have not been implemented yet.

---

## 3. LangGraph graph definition

**Why Actor-Critic-Boss lives here (in Silver layer):** Bronze does zero LLM calls by design and 
Gold is where cross-meeting reconciliation belongs, Silver is a layer that can generate
content with real hallucination risk (`synthesize_document`'s component lifecycle tags and ODCS contract drafts),
and every`synthesize_document` call is followed by Critic and Boss, no exceptions, no confidence threshold
that skips it.

**State:**

```
class SilverState(TypedDict):
    ingestion_date: str
    bronze_documents: list[dict]
    transcript_text: str
    generated_questions: list[GeneratedQuestion] # union of nodes 2+3; id/scope/target/requirement/question
    mentioned_components: list[MentionedComponentItem]  # name, status — drafted by node 2, used
                                                          # by generate_architecture_questions
                                                          # itself only; not threaded into
                                                          # synthesize_document (node 7) — the ADR
                                                          # prompt derives component status itself
                                                          # from the transcript + clarifications
    clarifications: list[ClarificationItem]     # id/scope/target/requirement/question, answer, status
    pending_questions: list[str]
    known_gold_components: list[dict]           # vestigial — no longer populated or read; see node 7
    documents: dict[str, str]                   # source_component -> synthesized ADR Markdown (Actor)
    document_versions: dict[str, int]           # source_component -> version write_document just wrote
                                                 # (node 10), read back by chunk_and_embed (node 11)
    critiques: dict[str, list[CritiqueItem]]    # source_component -> [{claim, supported, rationale}]
    boss_verdicts: dict[str, str]               # source_component -> "ok" | "needs_human_review"
    revision_attempted: dict[str, bool]         # source_component -> already retried once?
    active_sources: list[str]                   # sources synthesize/critic/boss are working on this pass
    redraft_only: list[str] | None              # set by boss_decide to redraft just these sources
    interrupt_origin: Literal["classify", "boss"]  # how ask_human should read its resume payload
```

No `human_answers` field: `ask_human`'s resume payload arrives directly as the return value of
`interrupt()` at the call site (LangGraph's own mechanism), so there's nothing to stage in state
ahead of time — a separate field would just be a second, redundant place for the same value to
live. `active_sources`/`redraft_only` exist because Actor-Critic-Boss operates per source
transcript, not on the whole batch at once: `redraft_only` lets a Boss-escalated retry redraft just
the flagged source instead of every source in the ingestion date.

**Nodes:**

The graph in eleven short steps — what each one does, and whether it costs an LLM call:

1. **`load_bronze`** — No LLM. Loads every transcript chunk for this date into one block of text.

2. **`generate_architecture_questions`** — 1 LLM call. Reads the transcript and drafts: which
   components are involved (and whether each is new/modified/removed/unchanged), which data
   contracts exist (just their name/producer/consumer/action, not full detail yet), and the
   architecture/ADR-level questions still open. Writes an audit copy to
   `output/ingestion_date=<date>/questions/`.

3. **`generate_data_contract_questions`** — 1 LLM call, skipped (free) if step 2 found no
   contracts. Takes exactly the contracts step 2 named and drafts the full completeness
   questions for each (schema, quality, versioning...). Writes its own audit copy to
   `output/ingestion_date=<date>/data_contract_questions/`.

4. **`classify_questions`** — 1 LLM call. Sorts every question from steps 2-3 into: the
   transcript already answers it, it's genuinely unknown, or a human needs to answer it.

5. **`route_after_classify`** — No LLM. If anything needs a human, go ask; otherwise skip ahead.

6. **`ask_human`** — No LLM. Pauses and asks all pending questions at once (capped separately for
   architecture vs. data-contract questions, see `app/config.py`). Resumes once answered.

7. **`synthesize_document`** — 1 LLM call per transcript. Writes the actual ADR, using only the
   transcript and the answered questions. Sections nothing supports are left as "not covered."

8. **`critic_document`** — 1 LLM call per transcript. Re-reads the ADR against the source and
   flags any claim that isn't actually backed up.

9. **`boss_decide`** — No LLM. A weak claim gets quietly downgraded. A real contradiction goes
   back to a human once (via `ask_human` again), then the ADR is redrafted for that one source.

10. **`write_document`** — No LLM. Saves the ADR to Postgres (a new version only if the content
    actually changed) and writes a copy to `output/ingestion_date=<date>/adr/`.

11. **`chunk_and_embed`** — No LLM. Embeds the ADR as one retrievable chunk. `END`.

**Cost per transcript:** up to 5 LLM calls (4 if no data contracts were found) — steps 2 and 4
run once for the whole batch, step 3 once per batch (or zero), steps 7 and 8 once per
transcript. Steps 1, 5, 6, 9, 10, 11 never call an LLM.

```
load_bronze ──► generate_architecture_questions ──► generate_data_contract_questions ──► classify_questions ──► route_after_classify ─┬─► synthesize_document ──► critic_document ──► boss_decide ─┬─► write_document ──► chunk_and_embed ──► END
                                                                                                                │                                                            │
                                                                                                                └─► ask_human ──(interrupt)──(resume)──► route_after_classify
                                                                                                                                                                             │
                                                                                                   (contradiction, first pass) └─► ask_human ──(interrupt)──(resume)──► synthesize_document [retry, this source only]
```

**Tracing:** every node above shows up as its own span in Logfire, nested under one span per
graph run — `agents/graph.py` sets LangGraph's built-in OpenTelemetry tracing (via its bundled
LangSmith SDK) to route through `logfire.configure()`'s global tracer provider before `langgraph`
itself is imported. No LangSmith account, exporter, or API key involved — just
`LOGFIRE_TOKEN` (optional; unset stays local-only, console output, no network calls).


## 7. Explicit non-goals

- **No point-in-time architecture diagrams yet.** `generate_architecture_questions`'s
  identity-matching (§3 node 2) is designed to compare a transcript against the architecture as it stood at that transcript's
  own date, but nothing in this project stores or versions architecture diagrams by date. Until
  that source exists, `architecture_diagram` stays empty on every run, which the prompt already
  treats as correct (a first-time description), not broken. Building that source is future,
  Gold-adjacent work, not part of Silver.
- **Gold doesn't exist yet.** No Gold tables, module, or migration exist in this codebase today —
  `_lookup_known_gold_components` (a best-effort, tolerate-missing-table lookup) was removed from
  `synthesize_document` for exactly this reason once the ADR prompt stopped needing it. Any
  cross-meeting reconciliation, a real component graph, or an authoritative version history is
  future, unscoped work — not a table sitting unused, an absence.
- **Silver's `version` is not component-version authority.** `silver_documents.version` only
  disambiguates "did this specific ADR's content change since last time" for one source
  transcript, via a content hash — it says nothing about a component's real-world version history
  across meetings. That reconciliation (if it ever exists) is Gold-level work Silver doesn't do or
  claim to do.
- **Full ADR coverage isn't the goal — honest partial coverage is.** `synthesize_document` fills
  exactly what the clarifications support and collapses everything else to the ADR prompt's own
  "not covered" notes — and both question-generation prompts are scoped to never draft a
  question about the sections nobody narrates in a meeting
  (`prompts/architecture_questions.jinja`/`data_contract_questions.jinja`, **Out
  of scope** / **Important distinction**).


