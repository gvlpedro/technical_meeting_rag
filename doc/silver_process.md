# Silver process is a clarification loop to store clarified documents in silver layer

Bronze holds raw, unreviewed transcript chunks. Silver's job is to turn one ingestion batch
(`bronze_documents` for a given `ingestion_date`) into **one clarified document per source
transcript**, shaped as `doc/clarification_template.md` itself — the same
document a human architect would fill with human/agent colaboration, filled in as far as the clarification loop's
answers allow, **not necessarily complete** — a partially filled template is the expected, correct
output, not a defect. Written as **structured Markdown** in Postgres.

**Idempotency:** at most one clarified document per source transcript. Re-running the process
upserts that transcript's `silver_documents` row in place (unique on `source_component`, refreshed
content) and replaces that transcript's `silver_chunks` rows rather than accumulating duplicates.

## 1. Storage

**In Postgres** — two tables:

```
silver_documents                       -- one row per source transcript's full clarified document
  id                 pk
  ingestion_date     date, not null, index
  source_component   varchar, not null, unique   -- e.g. "design_twitter_system_design_interview.en"
                                                   -- — matches bronze_documents.source_component;
                                                   -- unique so a re-run upserts this row instead of
                                                   -- duplicating it (idempotency, §1)
  content            text, not null              -- clarification_template.md, filled in as far as
                                                 -- the clarification loop's answers allow (§3
                                                 -- node 5); unanswered sections stay as the
                                                 -- template's own placeholders, not fabricated

silver_chunks                          -- silver_documents.content, chunked + embedded for retrieval
  id                 pk
  ingestion_date     date, not null, index
  source_component   varchar, not null      -- which silver_documents row this chunk came from; no
                                             -- FK, same flat pattern bronze_documents already uses
  content            text, not null      -- one chunk of that document
  embedding          vector(settings.embedding_dim)
```


---

## 2. Clarification loop: the decision rule

For each question in `langgraph_agent/clarification_questions.json` 
(multiple questions to build a completed ADR), the loop lands on exactly one outcome:

| Outcome | Meaning | Action |
|---|---|---|
| `answered` | The transcript states it, or implies it unambiguously. | Record the answer. No human involved. |
| `unknown` | The transcript doesn't say, and it plausibly never will be known from this meeting. | Record `answer = null`. **Do not ask.** |
| `needs_clarification` | The transcript doesn't say, but a human involved would plausibly know. | Batch into the single `interrupt()` call. |

If human doesn't know an answer during resume downgrades that question to `unknown` post hoc — never
re-asked.

The loop should refine the final understanding of the organization asking to clarify following points:

1. **Clarify ambiguities:** Ask for items that are not clear enough and require additional information to properly define each perspective.
2. **Clarify contradictions:** Ask for items that are inconsistent between the different perspectives and require clarification or resolution.
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
    clarifications: list[ClarificationItem]     # question, answer, status
    pending_questions: list[str]
    human_answers: dict[str, str]
    known_gold_components: list[dict]           # name, description, profile — best-effort lookup
    documents: dict[str, str]                   # source_component -> synthesized Markdown (Actor)
    critiques: dict[str, list[CritiqueItem]]    # source_component -> [{claim, supported, rationale}]
    boss_verdicts: dict[str, str]               # source_component -> "ok" | "needs_human_review"
    revision_attempted: dict[str, bool]         # source_component -> already retried once?
```

**Nodes:**

The graph reads like a short story, not a checklist: gather what was said, figure out what's still
fuzzy, ask a human about the fuzzy part only, write the clarified account, check that account
against its own sources, fix or escalate anything it can't back up, then save and index the result.

1. **`load_bronze`** — *Gather what was said.* Pulls every Bronze row for this `ingestion_date` and
   concatenates it into `transcript_text`. Pure retrieval, nothing interpreted yet.

2. **`classify_questions`** — One LLM call reads the whole transcript against every clarification 
   question at once, sorting each into `answered`,`unknown`, or `needs_clarification` to know what we need to clarify
    with a human.

3. **`route_after_classify`** — Sends the graph to ask a human.

4. **`ask_human`** — *Ask, once, only about what's genuinely unclear.* Pauses the graph with a
   single `interrupt()` carrying every pending question together, not one interruption per
   question, relying on LangGraph's own Postgres-backed checkpointer to survive the pause. Whatever
   comes back on resume folds into `clarifications`, and the graph re-routes.

5. **`synthesize_document`** — *Fill in the template, honestly.* One LLM call per source transcript
   writes directly into `doc/clarification_template.md`'s own structure — not a
   separate, simplified shape — using everything now known: the transcript, the clarifications, and
   a best-effort `SELECT` against Gold's `components` table. It fills whatever
   `clarification_questions.json`'s answers actually support: Motivation and Context, Affected
   Components (each tagged new / evolving / unchanged / unknown relative to what Gold already
   knows, with a one-line reason), Data Contracts as ODCS drafts,, Alternatives, Consequences, 
   Risks, and so on down the questions' coverage.

6. **`critic_document`** — *Check the document's own homework.* One LLM call per source transcript,
   always run, reads the freshly drafted document back against the same transcript and
   clarifications the Actor saw, and flags any claim that isn't actually backed up — telling apart a
   thin claim (already honestly marked `evidenced: false`) from one that outright contradicts the
   transcript. Only the second kind matters to what comes next.

7. **`boss_decide`** — *Decide what to do about the Critic's flags.* No LLM call — a rule, not a
   second opinion. A thin claim gets quietly downgraded to `unknown` and the document moves on. A
   real contradiction gets escalated to a human, once, through the same `ask_human` mechanism
   already built, then the document is redrafted just for that one source. If it's still flagged
   after that single retry, it gets downgraded and let through rather than looping forever.

8. **`write_document`** — *Make it official.* No LLM call: upserts the finished document into
   `silver_documents` and the clarification audit trail into `clarifications.json`, both keyed so a
   re-run overwrites cleanly instead of piling up duplicates.

9. **`chunk_and_embed`** — *Make it retrievable.* No LLM call: clears any previous chunks for this
   date, splits the document with the same chunker Bronze already uses, embeds each piece, and
   inserts the fresh `silver_chunks` rows. `END`.

Three LLM calls per source transcript in the clean case — one `classify_questions`
call shared by the whole batch, plus one `synthesize_document` and one `critic_document` call per
transcript. `boss_decide` never adds an LLM call, only a human round-trip when the Critic finds a
real contradiction, and never more than one retry per document.

```
load_bronze ──► classify_questions ──► route_after_classify ─┬─► synthesize_document ──► critic_document ──► boss_decide ─┬─► write_document ──► chunk_and_embed ──► END
                                                             │                                                            │
                                                             └─► ask_human ──(interrupt)──(resume)──► route_after_classify
                                                                                                                          │
                                                                              (contradiction, first pass) └─► ask_human ──(interrupt)──(resume)──► synthesize_document [retry, this source only]
```


## 7. Explicit non-goals

- **Gold's schema is decided, its process isn't.** Gold has two tables now — `adr` (evolution
  decisions) and `components` (each component's version history) — but structure-aware chunking,
  the component graph, and the actual "Pull request / versions" mechanism that populates and
  reconciles those tables stay unscoped. A table existing doesn't mean the process that fills it
  does.
- **No versioning authority from Silver.** The lifecycle tagging in the template's Affected
  Components section is a **proposal** against a best-effort, possibly-empty snapshot of Gold's
  `components` table — it is not the system of record. Only Gold's reconciliation process (not yet
  built, even though `adr`/`components` already exist as tables) may authoritatively decide a
  component's version history.
- **Full template coverage isn't the goal — honest partial coverage is.** `synthesize_document`
  fills exactly what `clarification_questions.json`'s questions can extract from a transcript
  (context, alternatives, trade-offs, constraints, component purpose/lifecycle, risks,
  assumptions, decision status).
