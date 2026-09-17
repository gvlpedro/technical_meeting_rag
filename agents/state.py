from typing import Literal, TypedDict

from app.config import settings

ClarificationStatus = Literal["answered", "unknown", "needs_clarification", "human_answered"]


class GeneratedQuestion(TypedDict):
    """One question as drafted by `generate_architecture_questions` or
    `generate_data_contract_questions` — see `agents.schemas.QuestionItem`. `id` is what
    `classify_questions` matches a classification back to, not question text. Both
    generation stages append into the same `generated_questions` list — `classify_questions`
    classifies their union in one pass."""

    id: str
    scope: str
    target: str
    requirement: str
    question: str


class MentionedComponentItem(TypedDict):
    """See `agents.schemas.MentionedComponent` — the Actor's own lifecycle read for a
    component the transcript names, drafted by `generate_architecture_questions`."""

    name: str
    status: str


class MentionedDataContractItem(TypedDict):
    """See `agents.schemas.MentionedDataContract` — a data contract
    `generate_architecture_questions` identified (name/producer/consumer/action only), fed
    into `generate_data_contract_questions` as its fixed `IDENTIFIED_DATA_CONTRACTS` input."""

    name: str
    producer: str
    consumer: str
    action: str


class ClarificationItem(TypedDict):
    id: str
    scope: str
    target: str
    requirement: str
    question: str
    answer: str | None
    status: ClarificationStatus


class BronzeRow(TypedDict):
    source_component: str
    content: str


class CritiqueItem(TypedDict):
    claim: str
    supported: bool
    rationale: str
    severity: Literal["low", "material"]


class SilverState(TypedDict):
    """See `doc/silver_process.md` §3 for the full rationale behind each field.

    Deviates from that document's illustrative state sketch in one way: there is no
    `human_answers` field. `ask_human`'s resume payload arrives directly as the
    return value of `interrupt()` at the call site (LangGraph's own mechanism), so
    there is nothing to stage in state ahead of time — storing it separately would
    just be a second, redundant place for the same value to live.
    """

    tenant: str  # which frontend tenant this run belongs to — threaded into every DB write/query
    # The logged-in username driving this run, if any — `write_document` stamps it onto every
    # synthesized ADR as a `**Authors:**` line (`agents.service.insert_authors_line`), then
    # `_persist_document_version` reads it straight back out of that same content into
    # `SilverDocument.authored_by`. `""` for a CLI/script/test run with no real user — see
    # `insert_authors_line`'s own docstring for why that's a safe no-op, not a placeholder.
    username: str
    # Whether `write_document` may persist this run's own `SilverDocument`/on-disk audit file,
    # and whether the graph continues on into Gold at all (`route_after_write_document`) — `True`
    # everywhere except the frontend's initial upload (`app/routers/frontend.py::
    # upload_transcription` passes `False`). The clarification loop's own draft (and any
    # "Regenerate ADR"/"Ask me more" refinement of it) is temporary until a human explicitly
    # clicks "Publish" (`finalize_document`); nothing before that click may create a Silver
    # version or a Gold fact. `SilverClarification` (the Q&A audit trail) is the one exception —
    # `write_document` always logs it regardless of `persist`, since `regenerate_document`/
    # `ask_more_questions` read it back to keep grounding a still-unpublished draft.
    persist: bool
    # Per-run override of settings.max_architecture_pending_questions/
    # max_data_contract_pending_questions (see _top_questions) — the frontend's "Input
    # transcription" tab lets a user set one shared number for both per upload; a run started
    # any other way (a script, a test) falls back to the settings default (see initial_state).
    max_architecture_pending_questions: int
    max_data_contract_pending_questions: int
    # Restricts this run's Bronze batch to exactly these source_components, instead of every
    # bronze_documents row that happens to share (tenant, ingestion_date) — see
    # `agents.service.load_bronze_rows`'s own docstring for why this matters: two unrelated
    # frontend uploads picking the same calendar date would otherwise get pooled into one
    # batch, cross-contaminating each other's questions/ADR. `None` (a script, a test, `make
    # clarify`) keeps the original "everything for this date" pooling — that one is
    # intentional, a real batch of same-day transcripts meant to be processed together.
    source_components: list[str] | None
    ingestion_date: str
    bronze_documents: list[BronzeRow]
    transcript_text: str
    generated_questions: list[GeneratedQuestion]  # union of both generation stages, this batch
    mentioned_components: list[MentionedComponentItem]  # drafted by generate_architecture_questions
    mentioned_data_contracts: list[MentionedDataContractItem]  # same stage; feeds the next one
    clarifications: list[ClarificationItem]
    pending_questions: list[str]
    documents: dict[str, str]  # source_component -> synthesized ADR Markdown (Actor)
    document_versions: dict[str, int]  # source_component -> version write_document just wrote
    critiques: dict[str, list[CritiqueItem]]  # source_component -> Critic's findings
    adr_scores: dict[str, int]  # source_component -> Critic's completeness_score (0-100)
    adr_unresolved_points: dict[str, list[str]]  # source_component -> Critic's unresolved_points
    boss_verdicts: dict[str, str]  # source_component -> "ok" | "needs_human_review"
    revision_attempted: dict[str, bool]  # source_component -> already retried once?
    active_sources: list[str]  # source_components synthesize/critic/boss are working on this pass
    redraft_only: list[str] | None  # set by boss_decide: redraft just these sources, not the batch
    interrupt_origin: Literal["classify", "boss"]  # how ask_human should interpret its resume payload
    # Gold (.tmp/gold_process_v5.md §1-2) — three more nodes appended after chunk_and_embed,
    # sharing this same graph run instead of a separately triggered pass. Neither field needs
    # to survive an `interrupt()`/resume cycle (Gold nodes never interrupt), but both live in
    # state rather than as node-local variables so a redraft's second pass through this graph
    # doesn't lose an earlier source's already-computed extraction.
    gold_extractions: dict[str, dict]  # source_component -> extract_gold_facts' GoldExtractionResult
    gold_entity_ids: dict[str, dict[str, str]]  # source_component -> {raw name -> resolved entity_id}


def initial_state(
    ingestion_date: str,
    tenant: str = "default",
    max_architecture_pending_questions: int | None = None,
    max_data_contract_pending_questions: int | None = None,
    source_components: list[str] | None = None,
    persist: bool = True,
    username: str = "",
) -> SilverState:
    return {
        "tenant": tenant,
        "persist": persist,
        "username": username,
        "max_architecture_pending_questions": (
            max_architecture_pending_questions
            if max_architecture_pending_questions is not None
            else settings.max_architecture_pending_questions
        ),
        "max_data_contract_pending_questions": (
            max_data_contract_pending_questions
            if max_data_contract_pending_questions is not None
            else settings.max_data_contract_pending_questions
        ),
        "source_components": source_components,
        "ingestion_date": ingestion_date,
        "bronze_documents": [],
        "transcript_text": "",
        "generated_questions": [],
        "mentioned_components": [],
        "mentioned_data_contracts": [],
        "clarifications": [],
        "pending_questions": [],
        "documents": {},
        "document_versions": {},
        "critiques": {},
        "adr_scores": {},
        "adr_unresolved_points": {},
        "boss_verdicts": {},
        "revision_attempted": {},
        "active_sources": [],
        "redraft_only": None,
        "interrupt_origin": "classify",
        "gold_extractions": {},
        "gold_entity_ids": {},
    }
