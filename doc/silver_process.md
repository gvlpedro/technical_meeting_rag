# Silver process is a clarification loop to store clarified ADRs in silver layer

Bronze holds raw, unreviewed transcript chunks. Silver's job is to turn one ingestion batch
(`bronze_documents` for a given `ingestion_date`) into **one clarified ADR per source
transcript**, shaped as `prompting/roles/common/adr_generator.jinja` produces it — filled in as
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

**In Postgres** — the source of truth, three tables. `generate_questions` (§3 node 2) also writes a
human-readable audit copy of its own output to `output/ingestion_date=<date>/questions/
<transcription>.json` — the one on-disk artifact in Silver, and it's exactly that, an artifact:
nothing downstream reads it back, `silver_clarifications` (below) stays the real record.

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
  content            text, not null              -- prompting/roles/common/adr_generator.jinja's
                                                 -- structure, filled in as far as the
                                                 -- clarification loop's answers allow (§3 node
                                                 -- 6); uncovered sections collapse to their
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
                                                 -- answer" — write_document (§3 node 6) inserts a
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
                                          -- retrieved whole, never split (§3 node 10)
  embedding          vector(settings.embedding_dim)

  UNIQUE (source_component, version)
```


---

## 2. Clarification loop: the decision rule

The question list itself isn't fixed anymore — it used to be a static, generic file
(`agents/clarification_questions.json`, deleted); now `generate_questions` (§3 node 2) drafts it
fresh per ingestion batch, reading the transcript against `prompting/roles/common/clarification_template.md` and
asking one specific question per identified component instead of one generic question that
implicitly spans all of them (`prompting/roles/common/clarification_questions.jinja` has the full
rationale and the process that drafts it). For each question in that drafted list, the loop lands
on exactly one outcome:

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
    generated_questions: list[GeneratedQuestion] # drafted fresh per batch — see node 2; id/scope/target/requirement/question
    mentioned_components: list[MentionedComponentItem]  # name, status — used by generate_questions
                                                          # itself only; no longer threaded into
                                                          # synthesize_document (node 6) — the ADR
                                                          # prompt derives component status itself
                                                          # from the transcript + clarifications
    clarifications: list[ClarificationItem]     # id/scope/target/requirement/question, answer, status
    pending_questions: list[str]
    known_gold_components: list[dict]           # vestigial — no longer populated or read; see node 6
    documents: dict[str, str]                   # source_component -> synthesized ADR Markdown (Actor)
    document_versions: dict[str, int]           # source_component -> version write_document just wrote
                                                 # (node 9), read back by chunk_and_embed (node 10)
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

The graph reads like a short story, not a checklist: gather what was said, figure out what's still
fuzzy, ask a human about the fuzzy part only, write the clarified account, check that account
against its own sources, fix or escalate anything it can't back up, then save and index the result.

1. **`load_bronze`** — *Gather what was said.* Pulls every Bronze row for this `ingestion_date` and
   concatenates it into `transcript_text`. Pure retrieval, nothing interpreted yet.

2. **`generate_questions`** — *Figure out what to even ask.* One LLM call reads the pooled
   transcript — and, when one exists, a Mermaid diagram of the architecture already known as of
   this transcript's own point in time — against `prompting/roles/common/clarification_template.md` (the
   `ARCHITECTURE_CHANGES` specification the prompt derives its required-information model from),
   and drafts two things fresh for this batch: `mentioned_components` (every component named,
   each with a `new`/`modified`/`removed`/`unchanged`/`unknown` status relative to the known
   architecture) and `questions` — each one a structured object (`id`, `scope`, `target`,
   `requirement`, `question`) rather than a bare string, so `classify_questions` can match a
   classification back to its question by `id` instead of by exact text. Questions span
   components, architecture-level entities (source, target, mechanism, producer/consumer), data
   contracts (identity, schema, quality, versioning/compatibility), change impact, and ADR-level
   decision fields, scoped to what the specification actually requires and the transcript
   actually leaves open — never a contract's own governance/bookkeeping. Full process and rules
   — a real Jinja template, not this
   project's own naive string-replace:
   `prompting/roles/common/clarification_questions.jinja`.
   `architecture_diagram` is **always empty today** —
   no source exists yet for "the architecture as of this specific transcript's date" (Gold doesn't
   version diagrams, §7) — so every run currently takes the prompt's own empty-architecture path:
   every component defaults to new, which is correct for a first-time description. The prompt
   already handles both cases; wiring a real diagram through is a future, separate change, not a
   gap in this node. Also writes a human-readable copy of the drafted result to
   `output/ingestion_date=<date>/questions/<transcription>.json` — one file per source transcript
   in the batch (same pooled result in each, since it's pooled across the whole `ingestion_date` —
   §2), named to mirror `input/transcriptions/ingestion_date=<date>/<transcription>.vtt`. Nothing
   downstream reads this file back; it exists to be inspected, not to be a second source of truth.

3. **`classify_questions`** — One LLM call reads the whole transcript against every question
   `generate_questions` just drafted, sorting each into `answered`, `unknown`, or
   `needs_clarification` to know what we need to clarify with a human.

4. **`route_after_classify`** — Sends the graph to ask a human.

5. **`ask_human`** — *Ask, once, only about what's genuinely unclear.* Pauses the graph with a
   single `interrupt()` carrying every pending question together, not one interruption per
   question, relying on LangGraph's own Postgres-backed checkpointer to survive the pause. Whatever
   comes back on resume folds into `clarifications`, and the graph re-routes.

6. **`synthesize_document`** — *Write the final ADR, honestly.* One LLM call per source
   transcript renders `prompting/roles/common/adr_generator.jinja`
   (`agents.prompts.build_adr_generation_prompt`) with the transcript and a flat question/answer
   list built from `clarifications` — `id`/`scope`/`target`/`requirement` are dropped, the ADR
   prompt only reads question text and its answer. Unlike the old `prompting/roles/common/clarification_template.md`
   -filling approach, this prompt derives component lifecycle status (`new`/`modified`/`removed`/
   `unchanged`) itself from the transcript and clarifications alone — `mentioned_components` and
   the Gold `components` lookup are no longer threaded in (see **State**, above). It fills exactly
   what the clarifications support — Context, Decision, Affected Components (each tagged with a
   status and a one-line reason), Affected Data Contracts as ODCS drafts — and collapses any
   section nothing supports to its prescribed one-sentence "not covered" note rather than a
   template placeholder; see the prompt file itself for the full component/data-contract inclusion
   rules.

7. **`critic_document`** — *Check the document's own homework.* One LLM call per source transcript,
   always run, reads the freshly drafted document back against the same transcript and
   clarifications the Actor saw, and flags any claim that isn't actually backed up — telling apart a
   thin claim (already honestly marked `evidenced: false`) from one that outright contradicts the
   transcript. Only the second kind matters to what comes next.

8. **`boss_decide`** — *Decide what to do about the Critic's flags.* No LLM call — a rule, not a
   second opinion. A thin claim gets quietly downgraded to `unknown` and the document moves on. A
   real contradiction gets escalated to a human, once, through the same `ask_human` mechanism
   already built, then the document is redrafted just for that one source. If it's still flagged
   after that single retry, it gets downgraded and let through rather than looping forever.

9. **`write_document`** — *Make it official, versioned.* No LLM call: hashes the finished ADR and
   compares it against the latest `silver_documents` row for this `source_component`. Same hash →
   overwrite that row in place (same `version`, idempotent re-run). Different hash → insert a new
   row at `version + 1`, leaving the older version's row untouched. Returns `document_versions`
   (`source_component -> version just written`) for `chunk_and_embed` to reuse. Separately appends
   this run's clarifications to `silver_clarifications` — one row per (source, question), inserted
   fresh every run rather than upserted, building a history instead of only the latest state. This
   node itself touches no disk — the one on-disk artifact in Silver is `generate_questions`' (node
   2).

10. **`chunk_and_embed`** — *Make it retrievable, one chunk per ADR.* No LLM call: computes a
    single embedding over the whole ADR — no token-splitting, an ADR is retrieved whole, never as
    a fragment — and upserts exactly the (`source_component`, `version`) pair `write_document` just
    wrote (using `document_versions`). Every other version's chunk row, for this source or any
    other, is left alone — the opposite of clearing every chunk for the `ingestion_date` up front,
    which would have destroyed the version history this node exists to keep. `END`.

Four LLM calls per source transcript in the clean case — `generate_questions` and
`classify_questions` each once, shared by the whole batch, plus one `synthesize_document` and one
`critic_document` call per transcript. `boss_decide` never adds an LLM call, only a human
round-trip when the Critic finds a real contradiction, and never more than one retry per document.

```
load_bronze ──► generate_questions ──► classify_questions ──► route_after_classify ─┬─► synthesize_document ──► critic_document ──► boss_decide ─┬─► write_document ──► chunk_and_embed ──► END
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

- **No point-in-time architecture diagrams yet.** `generate_questions`'s identity-matching (§3 node
  2) is designed to compare a transcript against the architecture as it stood at that transcript's
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
  "not covered" notes — and `generate_questions`'s prompt itself is scoped to never draft a
  question about the sections nobody narrates in a meeting
  (`prompting/roles/common/clarification_questions.jinja`, **Out of scope**).
