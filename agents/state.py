from typing import Literal, TypedDict

from app.config import settings

ClarificationStatus = Literal["answered", "unknown", "needs_clarification", "human_answered"]


class GeneratedQuestion(TypedDict):
    """One question drafted by `generate_architecture_questions` or
    `generate_data_contract_questions`. See `agents.shared.QuestionItem`.
    `classify_questions` uses `id` to match a classification back to this question.
    It does not match on the question text.
    Both generation stages add their questions to the same `generated_questions` list.
    `classify_questions` then classifies the combined list in one pass."""

    id: str
    scope: str
    target: str
    requirement: str
    question: str


class MentionedComponentItem(TypedDict):
    """See `agents.stages.architecture_questions.schemas.MentionedComponent`. This is the Actor's own read of the
    lifecycle status for a component the transcript names.
    `generate_architecture_questions` drafts it."""

    name: str
    status: str


class MentionedDataContractItem(TypedDict):
    """See `agents.stages.architecture_questions.schemas.MentionedDataContract`. This is a data contract that
    `generate_architecture_questions` identified. It only has the name, producer,
    consumer, and action. `generate_data_contract_questions` uses this as its fixed
    `IDENTIFIED_DATA_CONTRACTS` input."""

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
    """See `doc/silver_process.md` §3 for the full reasoning behind each field.

    This differs from that document's example state in one way. There is no
    `human_answers` field here. `ask_human`'s resume payload comes directly from the
    return value of `interrupt()` at the call site. That is LangGraph's own mechanism.
    So there is nothing to stage in state ahead of time. Storing it separately would
    just create a second, redundant place for the same value to live.
    """

    tenant: str  # The frontend tenant this run belongs to. We pass it into every DB write and query.
    # The logged-in username driving this run, if there is one. `write_document` stamps it
    # onto every synthesized ADR as a `**Authors:**` line (`agents.shared.insert_authors_line`).
    # Then `_persist_document_version` reads it back out of that same content into
    # `SilverDocument.authored_by`. This is `""` for a CLI, script, or test run with no real
    # user. See `insert_authors_line`'s own docstring for why that is a safe no-op, not a
    # placeholder.
    username: str
    # Whether `write_document` may save this run's own `SilverDocument` and on-disk audit file.
    # It also controls whether the graph continues on into Gold at all
    # (`route_after_write_document`). This is `True` everywhere except the frontend's initial
    # upload (`app/routers/frontend.py::upload_transcription` passes `False`).
    # The clarification loop's own draft, and any "Regenerate ADR" or "Ask me more" refinement
    # of it, stays temporary until a human clicks "Publish" (`finalize_document`).
    # Nothing before that click may create a Silver version or a Gold fact.
    # `SilverClarification` (the Q&A audit trail) is the one exception. `write_document`
    # always logs it, no matter what `persist` is, because `regenerate_document` and
    # `ask_more_questions` read it back to keep grounding a draft that is not published yet.
    persist: bool
    # Per-run override of settings.max_architecture_pending_questions and
    # settings.max_data_contract_pending_questions. See _top_questions.
    # The frontend's "Input transcription" tab lets a user set one shared number for both,
    # per upload. A run started any other way, such as a script or a test, falls back to the
    # settings default. See initial_state.
    max_architecture_pending_questions: int
    max_data_contract_pending_questions: int
    # Restricts this run's Bronze batch to exactly these source_components. Without this,
    # the batch would include every bronze_documents row that shares the same (tenant,
    # ingestion_date). See `agents.shared.load_bronze_rows`'s own docstring for why this
    # matters. Without the restriction, two unrelated frontend uploads that pick the same
    # calendar date would get pooled into one batch. That would cross-contaminate each
    # other's questions and ADR. `None` (used by a script, a test, or `make clarify`) keeps
    # the original "everything for this date" pooling. That pooling is intentional there: it
    # is a real batch of same-day transcripts meant to be processed together.
    source_components: list[str] | None
    ingestion_date: str
    bronze_documents: list[BronzeRow]
    transcript_text: str
    generated_questions: list[GeneratedQuestion]  # combined list from both generation stages, this batch
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
    # Gold (.tmp/gold_process_v5.md §1-2). Three more nodes run after chunk_and_embed, in the
    # same graph run, instead of a separately triggered pass. Neither field needs to survive
    # an `interrupt()` and resume cycle, because Gold nodes never interrupt. Both fields still
    # live in state, not as node-local variables, so a redraft's second pass through this graph
    # does not lose a source's extraction that was already computed.
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
