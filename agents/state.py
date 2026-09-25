from typing import Literal, TypedDict

from app.config import settings

ClarificationStatus = Literal["answered", "unknown", "needs_clarification", "human_answered"]


class GeneratedQuestion(TypedDict):
    """Architecture or data contract questions"""

    id: str
    scope: str
    target: str
    requirement: str
    question: str


class MentionedComponentItem(TypedDict):
    """Mentioned components"""

    name: str
    status: str


class MentionedDataContractItem(TypedDict):
    """Mentioned data contract items"""

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
    """See `doc/silver_process.md` for the full reasoning behind each field."""

    tenant: str
    username: str
    # Whether `write_document` may persist a SilverDocument/audit file and let the graph
    # continue into Gold — `True` everywhere except the frontend's initial upload, which
    # stays `False` until a human clicks "Publish" (`finalize_document`); `SilverClarification`
    # is the one exception, always logged regardless, since `regenerate_document`/
    # `ask_more_questions` need it to ground a still-unpublished draft.
    persist: bool
    # Per-run override of settings.max_architecture_pending_questions and
    # settings.max_data_contract_pending_questions. See _top_questions.
    # The frontend's "Input transcription" tab lets a user set one shared number for both,
    # per upload. A run started any other way, such as a script or a test, falls back to the
    # settings default. See initial_state.
    max_architecture_pending_questions: int
    max_data_contract_pending_questions: int
    # Restricts this run's Bronze batch to exactly these source_components instead of every
    # bronze_documents row sharing the same (tenant, ingestion_date)
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
