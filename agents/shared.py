"""Shared types and helper functions used by more than one pipeline stage, so no single stage
has to own and duplicate them."""

import re
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.state import BronzeRow
from app.config import settings
from db.models import BronzeDocument, SilverClarification, SilverDocument
from ingestion.bronze_documents_chunker import parse_ingestion_date

# Used by every stage that retries a bad LLM output: temperature 0 reliably repeats the same
# bad result, so a retry samples at this higher temperature instead, to actually get something
# different.
SHALLOW_RETRY_ATTEMPTS = 5
SHALLOW_RETRY_TEMPERATURE = 0.7

# --- Shared types --------------------------------------------------------------------------

ComponentStatus = Literal["new", "modified", "removed", "unchanged", "unknown"]
# `forward-update` = backward-compatible (only adds or loosens fields); `break-change` = not
# backward-compatible (removes/renames a field, or makes an optional one required).
#
# KNOWN LIMITATION: unlike `mentioned_components`, nothing checks this classification against
# the transcript — it's just the LLM's own judgment, so a real breaking change can get
# mislabeled `forward-update` and nothing downstream catches it.
ContractAction = Literal[
    "new", "forward-update", "break-change", "unchanged", "deprecated", "removed", "unknown"
]
QuestionScope = Literal[
    "metadata", "component", "architecture", "data_contract", "adr", "change_impact", "migration"
]


class QuestionItem(BaseModel):
    """One drafted clarification question, with a stable `id` so a later step (like
    `classify_questions`) can match it back without relying on the exact question text."""

    id: str
    scope: QuestionScope
    target: str
    requirement: str
    question: str


# --- Document-lifecycle helpers -------------------------------------------------------------


class NoBronzeDocumentsError(Exception):
    """Raised when there's nothing to clarify for this `ingestion_date`, so the graph fails
    loudly instead of quietly finishing with nothing written."""


def distinct_sources(bronze_documents: list[BronzeRow]) -> list[str]:
    seen: list[str] = []
    for row in bronze_documents:
        if row["source_component"] not in seen:
            seen.append(row["source_component"])
    return seen


def source_content(bronze_documents: list[BronzeRow], source_component: str) -> str:
    return " ".join(r["content"] for r in bronze_documents if r["source_component"] == source_component)


def insert_authors_line(document: str, username: str) -> str:
    """Adds a `**Authors:** <username>` line after the ADR's heading, in plain Python (never
    the LLM, since we already know the username) — must run after the Critic, which would
    otherwise flag this line as an unsupported claim, and does nothing when `username` is
    empty."""
    if not username:
        return document
    heading, _, rest = document.partition("\n")
    return f"{heading}\n\n**Authors:** {username}\n\n{rest.lstrip(chr(10))}"


_AUTHORS_LINE_RE = re.compile(r"^\*\*Authors:\*\*\s*(.+)$", re.MULTILINE)


def extract_authors_line(document: str) -> str:
    """Reads the username back out of a document's `**Authors:**` line — the reverse of
    `insert_authors_line`, returning `""` if there is no such line."""
    match = _AUTHORS_LINE_RE.search(document)
    return match.group(1).strip() if match else ""


def insert_source_line(document: str, source_component: str) -> str:
    """Adds a `**Source:** <source_component>` line after the ADR's heading, in plain Python
    (never the LLM, and never the version number, which isn't known yet), so a reader with
    only the downloaded `.md` file can still tell which transcript this ADR came from — must
    run after the Critic, same reason as `insert_authors_line`, and does nothing when
    `source_component` is empty."""
    if not source_component:
        return document
    heading, _, rest = document.partition("\n")
    return f"{heading}\n\n**Source:** {source_component}\n\n{rest.lstrip(chr(10))}"


_SOURCE_LINE_RE = re.compile(r"^\*\*Source:\*\*\s*(.+)$", re.MULTILINE)


def extract_source_line(document: str) -> str:
    """The inverse of `insert_source_line`. Returns `""` if the document has no such line —
    see `insert_source_line`."""
    match = _SOURCE_LINE_RE.search(document)
    return match.group(1).strip() if match else ""


def transcription_base_name(source_component: str) -> str:
    """Strips both the file extension and a trailing language code, so `"talk.en.vtt"`
    becomes `"talk"` — used wherever a stage names a file on disk after its source."""
    return Path(Path(source_component).stem).stem


def _write_json_audit_file(
    ingestion_date: str, bronze_documents: list[BronzeRow], subdir: str, payload: str
) -> None:
    """Writes one human-readable JSON audit file per source transcript in this batch, for a
    person to check — nothing downstream reads it back."""
    out_dir = Path(settings.output_dir) / f"ingestion_date={ingestion_date}" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in distinct_sources(bronze_documents):
        name = transcription_base_name(source)
        (out_dir / f"{name}.json").write_text(payload, encoding="utf-8")


_MIN_GROUNDABLE_NAME_LENGTH = 2


def name_appears_in_text(name: str, text_lower: str) -> bool:
    """Checks whether `name` appears in `text_lower` as a whole word, ignoring case, and
    rejects any name shorter than `_MIN_GROUNDABLE_NAME_LENGTH` — a plain substring check
    would wrongly match inside an unrelated word (e.g. "Order" inside "Reordering") or match
    almost anywhere for a 1-character name."""
    normalized = name.strip().lower()
    if len(normalized) < _MIN_GROUNDABLE_NAME_LENGTH:
        return False
    return re.search(rf"\b{re.escape(normalized)}\b", text_lower) is not None


def mentions_grounded_in_source(source_text: str, items: list[dict]) -> list[dict]:
    """Keeps only the items whose `name` actually appears in this one source's own transcript
    text, splitting the batch's pooled mentions back out per source-component (a name can
    rightly match more than one source; that's not a bug). KNOWN LIMITATION: since the
    mentions were drafted once from every source pooled together, a name the LLM reworded away
    from the transcript's own wording (e.g. "CO svc" reported as "Checkout Service") can end
    up attached to the wrong source, or to none — fixing that means tagging each mention with
    its source at generation time, out of scope here."""
    source_lower = source_text.lower()
    return [item for item in items if name_appears_in_text(item["name"], source_lower)]


# --- Reading Bronze/Silver for context (used by multiple stages + app/routers/frontend.py) --


async def load_bronze_rows(
    ingestion_date_str: str,
    db: AsyncSession,
    *,
    tenant: str = "default",
    source_components: list[str] | None = None,
) -> list[BronzeRow]:
    """Returns every `bronze_documents` row for this `(tenant, ingestion_date)` pair (raising
    `NoBronzeDocumentsError` if there are none), always filtered by tenant so two tenants'
    same-day meetings never mix, and narrowed to `source_components` when given so two
    unrelated uploads on the same date don't pool together either."""
    ingestion_date = parse_ingestion_date(ingestion_date_str)
    query = select(BronzeDocument.source_component, BronzeDocument.content).where(
        BronzeDocument.ingestion_date == ingestion_date, BronzeDocument.tenant == tenant
    )
    if source_components is not None:
        query = query.where(BronzeDocument.source_component.in_(source_components))
    result = await db.execute(query.order_by(BronzeDocument.id))
    rows: list[BronzeRow] = [
        {"source_component": r.source_component, "content": r.content} for r in result.all()
    ]
    if not rows:
        raise NoBronzeDocumentsError(_no_bronze_documents_message(ingestion_date_str))
    return rows


def _no_bronze_documents_message(ingestion_date_str: str) -> str:
    """The one error message shown when nothing was uploaded yet for this date."""
    return (
        f"No bronze_documents found for ingestion_date={ingestion_date_str!r} — check the date"
    )


async def bronze_content_for_source(session: AsyncSession, tenant: str, source_component: str) -> str:
    """Returns this source's raw transcript text, joined into one string — used by the
    frontend's "regenerate with feedback" endpoint, which runs outside any graph run."""
    result = await session.execute(
        select(BronzeDocument.content)
        .where(BronzeDocument.tenant == tenant, BronzeDocument.source_component == source_component)
        .order_by(BronzeDocument.id)
    )
    return " ".join(row[0] for row in result.all())


async def bronze_ingestion_date_for_source(
    session: AsyncSession, tenant: str, source_component: str
) -> date | None:
    """Returns this source's own `ingestion_date`, for the two callers that have no
    `SilverDocument` yet to read it from instead — returns `None` if this source has no
    Bronze rows at all."""
    result = await session.execute(
        select(BronzeDocument.ingestion_date)
        .where(BronzeDocument.tenant == tenant, BronzeDocument.source_component == source_component)
        .order_by(BronzeDocument.id)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def qa_pairs_for_source(session: AsyncSession, tenant: str, source_component: str) -> list[dict]:
    """Returns every clarification question and answer already recorded for this source, so
    "regenerate with feedback" can reuse the same resolved answers plus the reviewer's new
    feedback."""
    result = await session.execute(
        select(SilverClarification.question, SilverClarification.answer)
        .where(SilverClarification.tenant == tenant, SilverClarification.source_component == source_component)
        .order_by(SilverClarification.answered_at)
    )
    return [{"question": question, "answer": answer} for question, answer in result.all()]


async def latest_document_content(
    session: AsyncSession, tenant: str, source_component: str
) -> str | None:
    """Returns the most recent `SilverDocument` for this source (or `None` on a first-time
    run) — reads Silver, never Gold, since each layer here only ever reads from the layer
    right before it."""
    result = await session.execute(
        select(SilverDocument.content)
        .where(SilverDocument.tenant == tenant, SilverDocument.source_component == source_component)
        .order_by(SilverDocument.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _markdown_section(content: str, start_heading: str, end_heading: str) -> str:
    """Slices `content` from `start_heading` up to `end_heading` (or to the end if not
    found), as a plain string search rather than a Markdown parser — works because the ADR
    prompt always uses these exact heading names."""
    start = content.find(start_heading)
    if start == -1:
        return ""
    end = content.find(end_heading, start + len(start_heading))
    return (content[start:end] if end != -1 else content[start:]).strip()
