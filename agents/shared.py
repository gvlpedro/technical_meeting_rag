"""This is the cross-stage core. It holds types and helpers that more than one pipeline stage
needs. These live here instead of one stage owning them. If one stage owned them, other
stages would have to duplicate them, or import them in an awkward way.

Everything in `agents/stages/<stage>/` is specific to that one stage's own LLM call or calls.
Everything here is real shared infrastructure. This includes: reading Bronze and Silver for
context, the document-lifecycle helpers (`insert_authors_line` and `extract_authors_line`),
and a small set of Literal and Pydantic types that more than one stage's schema needs
(`ComponentStatus`, `ContractAction`, `QuestionScope`, `QuestionItem`).

See `agents/graph.py`'s own module docstring for how the ten pipeline stages and nodes fit
together. This module is the one piece every stage can depend on. Depending on it does not
count as cross-stage coupling.
"""

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

# This retry tuning is shared by every stage that retries at a nonzero temperature. They do
# this after a temperature=0 attempt shows a known failure pattern. This applies to
# `agents.stages.architecture_questions.service`, `agents.stages.data_contract_questions.
# service`, and `agents.graph.synthesize_document` for ADR generation. Temperature=0 makes a
# good run reliably reproducible. But it also reliably reproduces a bad run the same way. So
# each retry samples at this nonzero temperature instead. This is a way to break out of that
# fixed behavior. It is not a general "retry on any failure" policy.
SHALLOW_RETRY_ATTEMPTS = 5
SHALLOW_RETRY_TEMPERATURE = 0.7

# --- Shared types --------------------------------------------------------------------------

ComponentStatus = Literal["new", "modified", "removed", "unchanged", "unknown"]
# `forward-update` means the change is backward-compatible: it only adds fields or makes them
# optional. `break-change` means it removes or renames a field, or makes an optional field
# required. See prompts/data_contract_questions/questions.jinja PHASE 4 for the exact rule.
#
# KNOWN LIMITATION: `mentioned_components` gets checked word-for-word against the transcript by
# `agents.stages.architecture_questions.service._ungrounded_component_names`. This
# classification has no such check. It is only what the LLM decides by following the prompt. A
# change that quietly breaks something can look like "just adding detail" in the transcript.
# The LLM can then label it `forward-update` by mistake, and nothing downstream catches this
# before it ships as an authoritative ODCS spec. This is not fixed here. Fixing it would need a
# real schema-diff mechanism. That is out of scope for this pass.
ContractAction = Literal[
    "new", "forward-update", "break-change", "unchanged", "deprecated", "removed", "unknown"
]
QuestionScope = Literal[
    "metadata", "component", "architecture", "data_contract", "adr", "change_impact", "migration"
]


class QuestionItem(BaseModel):
    """One drafted clarification question. It carries a stable `id` (see each stage prompt's
    own QUESTION IDENTIFIERS section). This lets downstream steps, like
    `agents.graph.classify_questions`, match a classification back to its question. They do
    not need to rely on the question text being exactly the same. Both question-generation
    stages share this shape: `agents.stages.architecture_questions.schemas.
    ArchitectureQuestionListResult.questions` and `agents.stages.data_contract_questions.
    schemas.DataContractQuestionListResult.questions` both produce the same object shape."""

    id: str
    scope: QuestionScope
    target: str
    requirement: str
    question: str


# --- Document-lifecycle helpers -------------------------------------------------------------


class NoBronzeDocumentsError(Exception):
    """No `bronze_documents` rows exist for this `ingestion_date`. There is nothing to clarify.

    We raise this instead of letting the graph "succeed" with an empty `documents` list. If we
    did that, the graph would finish silently. Zero sources means every later node's
    per-source loop (`synthesize_document`, `critic_document`, `boss_decide`,
    `write_document`, `chunk_and_embed`) has nothing to loop over. So nothing gets written and
    nothing gets printed. This would look exactly like a normal, successful run, even though
    nothing happened.
    """


def distinct_sources(bronze_documents: list[BronzeRow]) -> list[str]:
    seen: list[str] = []
    for row in bronze_documents:
        if row["source_component"] not in seen:
            seen.append(row["source_component"])
    return seen


def source_content(bronze_documents: list[BronzeRow], source_component: str) -> str:
    return " ".join(r["content"] for r in bronze_documents if r["source_component"] == source_component)


def insert_authors_line(document: str, username: str) -> str:
    """Inserts a `**Authors:** <username>` line right after the ADR's own `# ADR — <title>`
    heading. This is plain, deterministic Python. We never leave this to the LLM. We already
    know the reviewer's identity from their session. It is not something we need to extract
    from the transcript. Asking the model to write it would only add a risk of hallucination,
    with no benefit.

    This must be called AFTER the Critic has already reviewed the document
    (`agents.graph.write_document`, `app/routers/frontend.py::regenerate_document`). Never
    call it before. The Critic checks every claim against `TRANSCRIPT` and `CLARIFICATIONS`.
    Neither one mentions who is running this session. If we inserted this line earlier, the
    Critic would flag it as a claim with no support, and downgrade it with a
    `[unknown — flagged by review]` marker.

    This function does nothing when `username` is empty or falsy. That happens on a CLI,
    script, or test run with no real logged-in user. In that case the document comes back
    exactly unchanged, byte for byte. So no existing golden-set or exact-match test output
    changes just because this function exists."""
    if not username:
        return document
    heading, _, rest = document.partition("\n")
    return f"{heading}\n\n**Authors:** {username}\n\n{rest.lstrip(chr(10))}"


_AUTHORS_LINE_RE = re.compile(r"^\*\*Authors:\*\*\s*(.+)$", re.MULTILINE)


def extract_authors_line(document: str) -> str:
    """The inverse of `insert_authors_line`. It reads the username back out of a document's
    own `**Authors:**` line. This gives `agents.graph._persist_document_version` exactly one
    source of truth for `SilverDocument.authored_by`: the content itself. Without this, a
    second value would be passed in separately, and that value could drift from what the
    document actually says. Returns `""` if the document has no such line. That happens on a
    CLI, script, or test run — see `insert_authors_line`."""
    match = _AUTHORS_LINE_RE.search(document)
    return match.group(1).strip() if match else ""


def transcription_base_name(source_component: str) -> str:
    """"real_time_delivery_architecture_at_twitter.en.vtt" becomes
    "real_time_delivery_architecture_at_twitter". This strips both the `.vtt` extension and
    the language-code suffix that `scripts/download_transcript.py` adds. It does this by
    taking `Path.stem` twice, once for each suffix. Every stage that names a file on disk
    after a source uses this. Examples: `agents.graph.write_document`'s ADR audit file, and
    `agents.stages.gold.service.current_architecture_diagram`'s node subtitle."""
    return Path(Path(source_component).stem).stem


def _write_json_audit_file(
    ingestion_date: str, bronze_documents: list[BronzeRow], subdir: str, payload: str
) -> None:
    """Writes `output/ingestion_date=<date>/<subdir>/<transcription>.json`. This writes one
    file per distinct source transcript in this batch. Each file gets the same pooled result.
    The pooling covers the whole ingestion_date, the same as `silver_clarifications` does
    elsewhere (`doc/silver_process.md` §2). This file is a human-readable audit copy that a
    person can check. Nothing downstream reads it back. Both question-generation stages' own
    service modules share this function."""
    out_dir = Path(settings.output_dir) / f"ingestion_date={ingestion_date}" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in distinct_sources(bronze_documents):
        name = transcription_base_name(source)
        (out_dir / f"{name}.json").write_text(payload, encoding="utf-8")


_MIN_GROUNDABLE_NAME_LENGTH = 2


def name_appears_in_text(name: str, text_lower: str) -> bool:
    """Checks whether `name` appears in `text_lower`. The caller has already lowercased
    `text_lower`. This check ignores case and respects word boundaries. A plain `in`
    substring check does not have these two guards. Both guards matter now, because the
    result gets saved to Postgres. It is no longer just feeding a short-lived retry
    heuristic.

    - **Word boundaries** (`\\b...\\b`, with the name regex-escaped): a plain
      `x in text_lower` check would let a short name match inside an unrelated word. For
      example, "Order" would match inside "Reordering" or "Orders" in a completely different
      source's transcript. That would silently attach a false mention to the wrong
      `silver_documents` row, with nothing to catch it later.
    - **Minimum length** (`_MIN_GROUNDABLE_NAME_LENGTH`): without this check, an empty or
      1-character `name` would match `"" in text_lower`, or match almost everywhere, for
      every source in the batch. We reject these short names outright instead of letting
      them "match" everywhere.

    This check is not perfect for a name that starts or ends with punctuation. That is
    because `\\b` needs a word/non-word change on each side. This is good enough for this
    domain's naming conventions (component, service, and contract names). It is not meant as
    a general-purpose text-matching tool.
    """
    normalized = name.strip().lower()
    if len(normalized) < _MIN_GROUNDABLE_NAME_LENGTH:
        return False
    return re.search(rf"\b{re.escape(normalized)}\b", text_lower) is not None


def mentions_grounded_in_source(source_text: str, items: list[dict]) -> list[dict]:
    """Returns every item, each a `MentionedComponentItem`- or `MentionedDataContractItem`-
    shaped dict, whose `name` appears word-for-word in this one source's own transcript
    content. The check ignores case and respects word boundaries — see
    `name_appears_in_text`.

    `mentioned_components` and `mentioned_data_contracts` get drafted once per
    `generate_architecture_questions_for_batch` call. That call reads the whole
    ingestion_date's pooled transcript text. The LLM's own output has no per-source split at
    all — see that function's docstring. This function is how `agents.graph.write_document`
    learns which of the batch's mentions actually belong to one specific
    `source_component`'s `silver_documents` row. It runs the same word-for-word check that
    `agents.stages.architecture_questions.service._ungrounded_component_names` already runs
    batch-wide, but the other way around, and scoped to one source's text instead of the
    pooled text. A name can rightly match in more than one source in the same batch. That is
    not a bug. It just means more than one transcript that day mentioned it.

    `write_document` calls this twice per source, once for components and once for
    contracts, against the same source's content. So the text gets lowercased twice per
    source. This cost is tiny next to the LLM calls already in the same pipeline run. It is
    not worth building an API that requires every caller to pre-lowercase its input — that
    would be an easy mistake waiting for the next caller who forgets to do it.

    KNOWN LIMITATION (not fixed here — see the `.tmp/refactor_silver_and_gold_process_v6.md`-
    era docs for the planned follow-up): `generate_architecture_questions_for_batch` pools
    every source's transcript into one LLM call before this grounding step ever runs. So the
    LLM's own output starts with no per-source attribution at all. This function rebuilds
    that attribution after the fact, by searching for substrings. That is a real problem, but
    a different one from the one fixed above. If the LLM rewords a name away from the
    source's own wording (for example, the transcript says "CO svc" but the LLM reports
    "Checkout Service"), that name matches nowhere, even with perfect word-boundary
    matching — a false negative. Meanwhile, a different source whose transcript happens to
    contain that exact reworded name absorbs it instead — a false positive. The real fix is
    to attach each mention to its source at generation time. This means either tagging each
    mention with its source, or running the architecture stage once per source. A better
    search after the fact will not fix this. That fix is out of scope for this pass.
    """
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
    """Returns every `bronze_documents` row for this `(tenant, ingestion_date)` pair, in the
    order they were inserted. Raises `NoBronzeDocumentsError` if there are none — see that
    class's docstring.

    We filter by `tenant` here, not only at ingestion time. Two different tenants can each
    have a meeting on the same calendar date. If we filtered by `ingestion_date` alone, we
    would silently pool both tenants' transcripts into one batch.

    When given, `source_components` narrows the result to exactly those source components.
    The frontend's "Input transcription" tab always passes the exact filenames it just
    ingested in this upload. This means a second, unrelated upload that happens to reuse the
    same calendar date never gets pooled with it. `None` keeps pooling every source under
    this date. A script, a test, or `make clarify` passes `None`. For that CLI-driven,
    whole-day workflow, pooling every source is the batch we want, not a mistake."""
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
    """Tells apart two different reasons why `bronze_documents` can be empty for a date. The
    first reason is a transcript file that was never ingested. This is the common case, and
    it is easy to miss: a file on disk is not the same thing as a row in the table. The
    second reason is that nothing exists at all — a typo in the date, or genuinely nothing
    uploaded yet. This function returns a different message for each case, instead of one
    generic message that reads the same either way."""
    transcripts_dir = Path(settings.input_dir) / "transcriptions" / f"ingestion_date={ingestion_date_str}"
    vtt_files = sorted(p.name for p in transcripts_dir.glob("*.vtt")) if transcripts_dir.is_dir() else []
    if vtt_files:
        return (
            f"Found {len(vtt_files)} transcript file(s) in {transcripts_dir} "
            f"({', '.join(vtt_files)}), but none are ingested into bronze_documents yet for "
            f"ingestion_date={ingestion_date_str!r}. A file on disk is not the same as a row in "
            f"the table — run `make ingestion DATE={ingestion_date_str}` first, then retry."
        )
    return (
        f"No bronze_documents found for ingestion_date={ingestion_date_str!r}, and no transcript "
        f"files exist at {transcripts_dir} either — nothing to ingest. Check the date, or place "
        f".vtt file(s) there first."
    )


async def bronze_content_for_source(session: AsyncSession, tenant: str, source_component: str) -> str:
    """Returns every `bronze_documents` chunk for this exact `(tenant, source_component)`
    pair, joined into one string. This is the same raw transcript text that
    `agents.graph.synthesize_document` reads through `source_content` inside the graph. Here
    we fetch it independently, for the frontend's "regenerate with feedback" endpoint. That
    endpoint runs the Actor again, outside of any graph run."""
    result = await session.execute(
        select(BronzeDocument.content)
        .where(BronzeDocument.tenant == tenant, BronzeDocument.source_component == source_component)
        .order_by(BronzeDocument.id)
    )
    return " ".join(row[0] for row in result.all())


async def bronze_ingestion_date_for_source(
    session: AsyncSession, tenant: str, source_component: str
) -> date | None:
    """Returns this source's own `ingestion_date`, exactly as `BronzeDocument` stores it. Two
    callers need this value and have no `SilverDocument` row to read it from instead. The
    first is the frontend's "ask me more" endpoint. It calls the question generators on their
    own, outside any graph run, just to name their own audit files under
    `output/ingestion_date=<date>/...` (formatted there with `.strftime("%Y%m%d")`). The
    second is `finalize_document`'s very first "Publish" for a source, because
    `_persist_document_version` takes a `date`, not a string. Returns `None` if this source
    has no bronze rows at all."""
    result = await session.execute(
        select(BronzeDocument.ingestion_date)
        .where(BronzeDocument.tenant == tenant, BronzeDocument.source_component == source_component)
        .order_by(BronzeDocument.id)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def qa_pairs_for_source(session: AsyncSession, tenant: str, source_component: str) -> list[dict]:
    """Returns every clarification question and answer already on record for this
    `(tenant, source_component)` pair, in the order they were answered. These come from
    `write_document`'s own append-only audit trail, `SilverClarification`. We rebuild that
    trail here so the frontend's "regenerate with feedback" endpoint can give the Actor the
    exact same resolved clarifications the original run used, plus one new entry for the
    reviewer's fresh feedback."""
    result = await session.execute(
        select(SilverClarification.question, SilverClarification.answer)
        .where(SilverClarification.tenant == tenant, SilverClarification.source_component == source_component)
        .order_by(SilverClarification.answered_at)
    )
    return [{"question": question, "answer": answer} for question, answer in result.all()]


async def latest_document_content(
    session: AsyncSession, tenant: str, source_component: str
) -> str | None:
    """Returns the content of the most recent existing `SilverDocument` for this
    `(tenant, source_component)` pair. This run uses it to answer "what did the previous ADR
    for this same source already say." It reads that answer from Silver's own version
    history. Returns `None` if this source has no prior version at all, which means this is a
    first-time run.

    This function reads Silver on purpose, and never Gold. Gold only ever mirrors what an
    already-written ADR states, one layer downstream. This pipeline follows a rule elsewhere
    too: each layer reads only from the layer right before it. For example,
    `agents.stages.gold.service` never reads `bronze_documents`, and the non-Gold stages
    never read `gold_evolution`. Keeping continuity between two runs of the same layer is a
    same-layer concern. It should use Silver's own last output. It is not a reason to break
    that rule."""
    result = await session.execute(
        select(SilverDocument.content)
        .where(SilverDocument.tenant == tenant, SilverDocument.source_component == source_component)
        .order_by(SilverDocument.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _markdown_section(content: str, start_heading: str, end_heading: str) -> str:
    """Slices `content` starting at `start_heading` (included) up to `end_heading`
    (excluded). If `end_heading` never appears, it slices to the end of the document. Returns
    `""` if `start_heading` itself is not found. This looks for the heading as a literal
    string. It is not a Markdown parser. This works reliably here because
    `prompts/adr_generation/generator.jinja`'s OUTPUT STRUCTURE fixes these exact heading
    strings. Both `agents.stages.architecture_questions.service.
    previous_architecture_context` and `agents.stages.adr_generation.service`'s own diagram
    extractors use this. It is the one Markdown-slicing tool both stages need."""
    start = content.find(start_heading)
    if start == -1:
        return ""
    end = content.find(end_heading, start + len(start_heading))
    return (content[start:end] if end != -1 else content[start:]).strip()
