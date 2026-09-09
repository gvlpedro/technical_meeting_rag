from typing import Literal, TypedDict

ClarificationStatus = Literal["answered", "unknown", "needs_clarification", "human_answered"]


class GeneratedQuestion(TypedDict):
    """One question as drafted by `generate_questions` — see `agents.schemas.QuestionItem`.
    `id` is what `classify_questions` matches a classification back to, not question text."""

    id: str
    scope: str
    target: str
    requirement: str
    question: str


class MentionedComponentItem(TypedDict):
    """See `agents.schemas.MentionedComponent` — the Actor's own lifecycle read for a
    component the transcript names, threaded into `synthesize_document`'s prompt below."""

    name: str
    status: str


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


class GoldComponentSnapshot(TypedDict):
    name: str
    description: str
    profile: str


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

    ingestion_date: str
    bronze_documents: list[BronzeRow]
    transcript_text: str
    generated_questions: list[GeneratedQuestion]  # drafted fresh per batch — see generate_questions
    mentioned_components: list[MentionedComponentItem]  # same batch — fed into synthesize_document
    clarifications: list[ClarificationItem]
    pending_questions: list[str]
    known_gold_components: list[GoldComponentSnapshot]
    documents: dict[str, str]  # source_component -> synthesized ADR Markdown (Actor)
    document_versions: dict[str, int]  # source_component -> version write_document just wrote
    critiques: dict[str, list[CritiqueItem]]  # source_component -> Critic's findings
    boss_verdicts: dict[str, str]  # source_component -> "ok" | "needs_human_review"
    revision_attempted: dict[str, bool]  # source_component -> already retried once?
    active_sources: list[str]  # source_components synthesize/critic/boss are working on this pass
    redraft_only: list[str] | None  # set by boss_decide: redraft just these sources, not the batch
    interrupt_origin: Literal["classify", "boss"]  # how ask_human should interpret its resume payload


def initial_state(ingestion_date: str) -> SilverState:
    return {
        "ingestion_date": ingestion_date,
        "bronze_documents": [],
        "transcript_text": "",
        "generated_questions": [],
        "mentioned_components": [],
        "clarifications": [],
        "pending_questions": [],
        "known_gold_components": [],
        "documents": {},
        "document_versions": {},
        "critiques": {},
        "boss_verdicts": {},
        "revision_attempted": {},
        "active_sources": [],
        "redraft_only": None,
        "interrupt_origin": "classify",
    }
