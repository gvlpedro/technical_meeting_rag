"""Reusable Silver logic — callable directly (CLI scripts, unit tests) without needing
the full LangGraph state shape, graph, or checkpointer. Mirrors `ingestion/service.py`'s
split: `agents/graph.py`'s nodes are thin wrappers calling the functions here; `make
questions`/`make ingestion` and this module's own tests call them directly.

Question generation is two sequential stages, not one: `generate_architecture_questions_for_batch`
(components, ADR, data-contract *identification*) followed by
`generate_data_contract_questions_for_batch` (full ODCS-completeness questions for exactly the
contracts the first stage identified). Splitting them fixed a real, observed failure mode: a
single combined pass regularly under-covered data contracts even when component/architecture
coverage was thorough — see `testing_arch_questions_acb/` and `testing_data_contract_questions_acb/`
for the two stages' own golden-set tests.
"""

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.prompts import (
    build_architecture_question_generation_prompt,
    build_data_contract_question_generation_prompt,
)
from agents.schemas import ArchitectureQuestionListResult, DataContractQuestionListResult
from agents.state import BronzeRow, MentionedDataContractItem
from agents.template import load_architecture_template, load_data_contract_template, load_json_response
from app.config import settings
from db.models import BronzeDocument, SilverClarification, SilverDocument
from ingestion.bronze_documents_chunker import parse_ingestion_date
from llm import router

SHALLOW_RETRY_ATTEMPTS = 5
SHALLOW_RETRY_TEMPERATURE = 0.7
# Floor for the data-contract stage's own shallowness check (see
# `_looks_shallow_data_contracts`) — ODCS completeness spans several distinct categories per
# contract (identity, schema, quality/SLA, servers, support, versioning), so a healthy run
# should draft noticeably more than one question per contract.
MIN_QUESTIONS_PER_CONTRACT = 5


class NoBronzeDocumentsError(Exception):
    """No `bronze_documents` rows exist for this `ingestion_date` — nothing to clarify.

    Raised instead of letting the graph "succeed" with empty `documents`, which would
    otherwise complete silently: zero sources means every later node's per-source loop
    (`synthesize_document`, `critic_document`, `boss_decide`, `write_document`,
    `chunk_and_embed`) has nothing to iterate over, so nothing gets written and
    nothing gets printed — indistinguishable from a real, unremarkable success.
    """


def distinct_sources(bronze_documents: list[BronzeRow]) -> list[str]:
    seen: list[str] = []
    for row in bronze_documents:
        if row["source_component"] not in seen:
            seen.append(row["source_component"])
    return seen


def source_content(bronze_documents: list[BronzeRow], source_component: str) -> str:
    return " ".join(r["content"] for r in bronze_documents if r["source_component"] == source_component)


def transcription_base_name(source_component: str) -> str:
    """"real_time_delivery_architecture_at_twitter.en.vtt" ->
    "real_time_delivery_architecture_at_twitter" — strips both the `.vtt` extension
    and the language-code suffix `scripts/download_transcript.py` adds, by taking
    `Path.stem` twice (once per suffix). Public: also used by `agents.graph.write_document`
    to name the on-disk ADR audit file the same way the audit-file writers below name
    their own."""
    return Path(Path(source_component).stem).stem


def _write_json_audit_file(
    ingestion_date: str, bronze_documents: list[BronzeRow], subdir: str, payload: str
) -> None:
    """Writes `output/ingestion_date=<date>/<subdir>/<transcription>.json` — one file per
    distinct source transcript in this batch, the same pooled result in each (pooled across
    the whole ingestion_date, same as `silver_clarifications` elsewhere —
    `doc/silver_process.md` §2). An inspectable, human-readable audit copy; nothing
    downstream reads it back."""
    out_dir = Path(settings.output_dir) / f"ingestion_date={ingestion_date}" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in distinct_sources(bronze_documents):
        name = transcription_base_name(source)
        (out_dir / f"{name}.json").write_text(payload, encoding="utf-8")


def _looks_shallow(result: ArchitectureQuestionListResult) -> bool:
    """Heuristic, not a hard rule — catches a specific failure mode observed in practice:
    the Actor correctly identifies every component (right names, right lifecycle status)
    but only drafts the most obvious per-component questions, then stops before ever
    reaching architecture- or ADR-level ones for the same transcript. Either signal is
    enough to suspect it: too few questions for how many components were identified, or
    every single question sharing one scope despite more than one component being in play.
    """
    if not result.mentioned_components:
        return False  # nothing to judge shallowness against
    if len(result.questions) < 2 * len(result.mentioned_components):
        return True
    scopes = {q.scope for q in result.questions}
    return len(scopes) == 1 and len(result.mentioned_components) > 1


_MIN_GROUNDABLE_NAME_LENGTH = 2


def name_appears_in_text(name: str, text_lower: str) -> bool:
    """Case-insensitive, word-boundary-safe check for whether `name` appears in `text_lower`
    (already lowercased by the caller). Two guards a plain `in` substring check doesn't have,
    both load-bearing now that the result is persisted to Postgres instead of only feeding a
    transient retry heuristic:

    - **Word boundaries** (`\\b...\\b`, name regex-escaped): a bare `x in text_lower` check
      would let a short name match *inside* an unrelated word — "Order" would "ground" against
      "Reordering"/"Orders" in a completely different source's transcript, silently attaching a
      false-positive mention to the wrong `silver_documents` row with nothing to catch it later.
    - **Minimum length** (`_MIN_GROUNDABLE_NAME_LENGTH`): an empty or 1-character `name` would
      otherwise satisfy `"" in text_lower` (or match everywhere) for *every* source in the
      batch — rejected outright instead of "grounding" everywhere.

    Not bulletproof for a name that itself starts/ends in punctuation (`\\b` needs a
    word/non-word transition on each side) — acceptable for this domain's naming conventions
    (component/service/contract names), not a general-purpose text-matching primitive.
    """
    normalized = name.strip().lower()
    if len(normalized) < _MIN_GROUNDABLE_NAME_LENGTH:
        return False
    return re.search(rf"\b{re.escape(normalized)}\b", text_lower) is not None


def _ungrounded_component_names(result: ArchitectureQuestionListResult, transcript: str) -> list[str]:
    """Every `mentioned_components` name that doesn't appear in the transcript verbatim
    (case-insensitive, word-boundary-safe — see `name_appears_in_text`) — a hallucinated or
    paraphrased component name, not something the transcript actually said. Same deterministic
    check `testing_arch_questions_acb`'s golden-set test already runs on the side, moved into
    production so a real `make questions`/`make clarify` run gets the same escape-hatch retry
    the test does, not just a post-hoc failure report."""
    transcript_lower = transcript.lower()
    return [c.name for c in result.mentioned_components if not name_appears_in_text(c.name, transcript_lower)]


def mentions_grounded_in_source(source_text: str, items: list[dict]) -> list[dict]:
    """Every item (a `MentionedComponentItem`/`MentionedDataContractItem`-shaped dict) whose
    `name` appears verbatim (case-insensitive, word-boundary-safe — see
    `name_appears_in_text`) in this one source's own transcript content.

    `mentioned_components`/`mentioned_data_contracts` are drafted once per `generate_
    architecture_questions_for_batch` call, over the whole ingestion_date's *pooled* transcript
    text — there is no per-source split in the LLM's own output (see that function's docstring).
    This is how `write_document` learns which of the batch's mentions actually belong to a
    specific `source_component`'s `silver_documents` row: the same verbatim-grounding check
    `_ungrounded_component_names` already runs batch-wide, inverted and scoped to one source's
    text instead of the pooled one. A name can legitimately ground in more than one source
    within the same batch — that's not a bug, it means more than one transcript that day
    mentioned it.

    `write_document` calls this twice per source (once for components, once for contracts)
    against the same source's content, so it's lowercased twice per source — negligible next to
    the LLM calls already in the same pipeline run, not worth an API that requires callers to
    pre-lowercase (a correctness footgun waiting for the next caller who forgets to).

    KNOWN LIMITATION (not fixed here — see `.tmp/refactor_silver_and_gold_process_v6.md`-era
    docs for the follow-up): `generate_architecture_questions_for_batch` pools every source's
    transcript into one LLM call *before* this grounding step ever runs, so the LLM's own
    output carries no per-source attribution to begin with — this function reconstructs it
    after the fact by substring search, which is a real but different failure mode from the
    one just fixed above: a name the LLM canonicalized away from a source's own wording (e.g.
    transcript says "CO svc", LLM reports "Checkout Service") grounds nowhere (false negative)
    even with perfect word-boundary matching, while a different source whose transcript happens
    to contain that literal canonical string absorbs it instead (false positive). The real fix
    is per-source attribution at generation time (tag each mention with its source, or run the
    architecture stage per source), not a better post-hoc search — out of scope for this pass.
    """
    source_lower = source_text.lower()
    return [item for item in items if name_appears_in_text(item["name"], source_lower)]


def _problem_count(result: ArchitectureQuestionListResult, transcript: str) -> int:
    """0, 1, or 2 — how many of the two retry triggers (shallow, ungrounded) a result has.
    Used to rank candidates across retry attempts: fewer problems always wins, and among
    equally-problematic (or equally clean) candidates, more questions wins as a tie-break."""
    return int(_looks_shallow(result)) + int(bool(_ungrounded_component_names(result, transcript)))


def _looks_shallow_data_contracts(
    result: DataContractQuestionListResult, mentioned_data_contracts: list[MentionedDataContractItem]
) -> bool:
    """Same rationale as `_looks_shallow`, scoped to the data-contract stage: too few
    questions relative to how many contracts were handed to it suggests some contracts got
    skipped or only superficially covered, not that they were genuinely already complete."""
    if not mentioned_data_contracts:
        return False
    return len(result.questions) < MIN_QUESTIONS_PER_CONTRACT * len(mentioned_data_contracts)


def _ungrounded_contract_targets(
    result: DataContractQuestionListResult, mentioned_data_contracts: list[MentionedDataContractItem]
) -> list[str]:
    """Every drafted question's `target` must match a contract this stage was actually given
    — this stage never discovers a new contract on its own (see `data_contract_questions.
    jinja`'s PHASE 12 — NO INVENTION); a target that doesn't match is either an invented
    contract or a component name leaking in where a contract name belongs."""
    known = {c["name"].lower() for c in mentioned_data_contracts}
    return sorted({q.target for q in result.questions if q.target.lower() not in known})


def _problem_count_data_contracts(
    result: DataContractQuestionListResult, mentioned_data_contracts: list[MentionedDataContractItem]
) -> int:
    return int(_looks_shallow_data_contracts(result, mentioned_data_contracts)) + int(
        bool(_ungrounded_contract_targets(result, mentioned_data_contracts))
    )


async def load_bronze_rows(
    ingestion_date_str: str,
    db: AsyncSession,
    *,
    tenant: str = "default",
    source_components: list[str] | None = None,
) -> list[BronzeRow]:
    """Every `bronze_documents` row for this `(tenant, ingestion_date)`, in insertion order.
    Raises `NoBronzeDocumentsError` if there are none — see that class's docstring.

    `tenant` filters here, not just at ingestion time: two tenants can both have a meeting on
    the same calendar date, and `ingestion_date` alone would silently pool both tenants'
    transcripts into one batch.

    `source_components`, when given, narrows further to exactly those source_components —
    the frontend's "Input transcription" tab always passes the exact filenames it just
    ingested in THIS upload, so a second, unrelated upload that happens to reuse the same
    calendar date never gets pooled with it. `None` (a script, a test, `make clarify`) keeps
    pooling every source under this date, which for that CLI-driven, whole-day-of-transcripts
    workflow is the intended batch, not an accident."""
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
    """Distinguishes the two ways `bronze_documents` can be empty for a date — a
    transcript file that was never ingested (the common, easy-to-miss case: the file
    on disk is not the same thing as a row in the table) versus nothing existing at
    all (a typo'd date, or genuinely nothing uploaded yet) — instead of one generic
    message that reads the same either way."""
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
    """Every `bronze_documents` chunk for this exact `(tenant, source_component)`, joined —
    the same raw transcript text `synthesize_document` reads via `source_content` inside the
    graph, fetched independently here for the frontend's "regenerate with feedback" endpoint,
    which runs the Actor again outside any graph run."""
    result = await session.execute(
        select(BronzeDocument.content)
        .where(BronzeDocument.tenant == tenant, BronzeDocument.source_component == source_component)
        .order_by(BronzeDocument.id)
    )
    return " ".join(row[0] for row in result.all())


async def bronze_ingestion_date_for_source(
    session: AsyncSession, tenant: str, source_component: str
) -> str | None:
    """This source's own `ingestion_date`, formatted the same `YYYYMMDD` way
    `generate_architecture_questions_for_batch`/`generate_data_contract_questions_for_batch`
    take it — needed by the frontend's "ask me more" endpoint, which calls both standalone
    (outside any graph run, so there is no `state["ingestion_date"]` to read) purely to name
    their own audit files under `output/ingestion_date=<date>/...`. `None` if this source has
    no bronze rows at all."""
    result = await session.execute(
        select(BronzeDocument.ingestion_date)
        .where(BronzeDocument.tenant == tenant, BronzeDocument.source_component == source_component)
        .order_by(BronzeDocument.id)
        .limit(1)
    )
    ingestion_date = result.scalar_one_or_none()
    return ingestion_date.strftime("%Y%m%d") if ingestion_date else None


async def qa_pairs_for_source(session: AsyncSession, tenant: str, source_component: str) -> list[dict]:
    """Every clarification question/answer already on record for this `(tenant,
    source_component)`, in the order they were answered — `write_document`'s own append-only
    audit trail (`SilverClarification`), reconstructed here so the frontend's "regenerate with
    feedback" endpoint can hand the Actor the exact same resolved clarifications the original
    run used, plus one new entry for the reviewer's fresh feedback."""
    result = await session.execute(
        select(SilverClarification.question, SilverClarification.answer)
        .where(SilverClarification.tenant == tenant, SilverClarification.source_component == source_component)
        .order_by(SilverClarification.answered_at)
    )
    return [{"question": question, "answer": answer} for question, answer in result.all()]


async def latest_document_content(
    session: AsyncSession, tenant: str, source_component: str
) -> str | None:
    """The content of the most recent existing `SilverDocument` for this
    `(tenant, source_component)` — this run's own continuity anchor for "what did the previous
    ADR for this same source already say," read from Silver's own version history. `None` if
    this source has no prior version at all (a first-time run).

    Deliberately reads Silver, never Gold, for this: Gold only ever mirrors what an already-
    written ADR states, one layer downstream — see the "each layer consumes only from the one
    before it" discipline this pipeline holds elsewhere (`agents/gold_service.py` never reads
    `bronze_documents`, `agents/service.py`/the non-Gold nodes of `agents/graph.py` never read
    `gold_evolution`). Continuity across two runs of the SAME layer is a same-layer concern —
    Silver's own last output — not a reason to bend that discipline."""
    result = await session.execute(
        select(SilverDocument.content)
        .where(SilverDocument.tenant == tenant, SilverDocument.source_component == source_component)
        .order_by(SilverDocument.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _markdown_section(content: str, start_heading: str, end_heading: str) -> str:
    """Slice `content` from `start_heading` (inclusive) up to `end_heading` (exclusive), or to
    the end of the document if `end_heading` never appears. `""` if `start_heading` itself isn't
    found. A literal-string heading match, not a Markdown parser — reliable here because
    `prompts/adr_generator.jinja`'s OUTPUT STRUCTURE fixes these exact heading strings."""
    start = content.find(start_heading)
    if start == -1:
        return ""
    end = content.find(end_heading, start + len(start_heading))
    return (content[start:end] if end != -1 else content[start:]).strip()


def _strip_diagram_colors(diagram: str) -> str:
    """Drops every `classDef`/`class` styling line from a Mermaid diagram string. A
    color class (see `prompts/adr_generator.jinja`'s DIAGRAM COLOR CODING section) marks what
    changed in the ONE ADR that drew it — it must never survive into a later document as if it
    were still true. Used by `previous_target_architecture_diagram`, whose output seeds a new
    ADR's "## 2. Previous Architecture": that section is always a plain, colorless snapshot,
    never the diff-coloring from whichever earlier ADR's "## 3. Target Architecture" this
    diagram came from."""
    lines = [line for line in diagram.splitlines() if not line.strip().startswith(("classDef", "class "))]
    return "\n".join(lines).strip()


def previous_target_architecture_diagram(previous_adr_content: str | None) -> str:
    """Pulls the ` ```mermaid ` fence out of a previous ADR's own "## 3. Target Architecture"
    section — this run's "Previous Architecture" is exactly what the last run's "Target
    Architecture" already was, minus any color coding (see `_strip_diagram_colors`). `""` if
    there is no previous ADR, or its Target Architecture section had no diagram (a first-time
    run, or one where nothing about the architecture was ever confirmed)."""
    if not previous_adr_content:
        return ""
    section = _markdown_section(previous_adr_content, "## 3. Target Architecture", "## 4. Affected Components")
    if "```mermaid" not in section:
        return ""
    diagram = section.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    return _strip_diagram_colors(diagram)


def own_previous_architecture_diagram(adr_content: str | None) -> str:
    """Pulls the ` ```mermaid ` fence out of an ADR's **own** "## 2. Previous Architecture"
    section — for redrafting the SAME still-unapproved draft (the frontend's "regenerate with
    feedback" endpoint), not for starting a new change from an already-approved one. That
    endpoint's `previous` is this very draft's latest `SilverDocument` row, already persisted by
    `write_document` before any human reviewed it — reading its §3 Target Architecture (via
    `previous_target_architecture_diagram`) would wrongly promote this unapproved draft's own
    target into a fabricated "previous" state for a change nothing has finalized yet. Reading its
    §2 instead reproduces exactly what this draft already established as prior architecture,
    unchanged by the extra feedback. `""` if there is no ADR (first-time run) or its §2 had no
    diagram (nothing about the prior architecture is established either).

    Also runs `_strip_diagram_colors` — §2 should never carry a color class in the first place,
    but this is the same belt-and-suspenders guarantee `previous_target_architecture_diagram`
    gives its own output, in case an earlier generation slipped one in anyway."""
    if not adr_content:
        return ""
    section = _markdown_section(adr_content, "## 2. Previous Architecture", "## 3. Target Architecture")
    if "```mermaid" not in section:
        return ""
    diagram = section.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    return _strip_diagram_colors(diagram)


def previous_architecture_context(previous_adr_content: str | None) -> str:
    """The previous ADR's Target Architecture diagram plus its Affected Components table,
    combined — this run's own `KNOWN_ARCHITECTURE` input for
    `generate_architecture_questions_for_batch`. Gives the LLM both the diagram (relationships)
    and the named components with their last confirmed status, so it can classify `new` vs.
    `unchanged` against something real instead of guessing with no anchor at all
    (`prompts/architecture_questions.jinja`'s own STATUS RULES). `""` if there is no previous
    ADR for this source (a first-time run)."""
    if not previous_adr_content:
        return ""
    return _markdown_section(
        previous_adr_content, "## 3. Target Architecture", "## 5. Affected Data Contracts"
    )


async def generate_architecture_questions_for_batch(
    ingestion_date_str: str, bronze_documents: list[BronzeRow], architecture_diagram: str = ""
) -> ArchitectureQuestionListResult:
    """Question-generation stage 1 of 2 (`doc/silver_process.md` §3 node 2): one LLM call
    reads the pooled transcript against `prompts/architecture_template.md` and
    drafts a fresh, transcript-specific question list covering every component and
    architecture/ADR-level gap the transcript establishes but leaves incomplete, plus
    *identifies* (never fully specifies) every data contract in play — see
    `prompts/architecture_questions.jinja`. Also returns `mentioned_components`
    (each with its own lifecycle `status`) and `mentioned_data_contracts` (name/producer/
    consumer/action) — the latter feeds directly into
    `generate_data_contract_questions_for_batch`.

    No DB session, no LangGraph state — just `bronze_documents` in, the parsed result out
    (and the `output/ingestion_date=<date>/questions/<transcription>.json` audit file written
    as a side effect). Callable directly from a script or a unit test with `bronze_documents`
    built by hand; `db/session.py` never enters the picture.

    `architecture_diagram` is empty only for a source with no prior ADR at all — `agents.graph.
    generate_architecture_questions` passes `agents.service.previous_architecture_context`'s
    output here, built from the previous `SilverDocument` for the same source (never from
    Gold — see that function's own docstring). The prompt's own Jinja
    `{% if known_architecture %}` already treats an empty value as "no prior architecture
    known," a correct, expected case for a source's first-ever run.
    """
    transcript_text = " ".join(row["content"] for row in bronze_documents)
    template_text = load_architecture_template()
    messages = build_architecture_question_generation_prompt(
        template_text, transcript_text, architecture_diagram=architecture_diagram
    )
    # temperature=0: this prompt is a mandatory, systematic checklist (see its own FINAL
    # SELF-CHECK section), not a creative-writing task — the default sampling temperature was
    # letting the same transcript sometimes get full coverage and sometimes skip parts of it
    # entirely, run to run, for no reason tied to the transcript itself. reasoning_effort=
    # "none" is required alongside it for a reasoning-locked model (e.g. gpt-5.6-sol/terra):
    # those models otherwise reject any temperature other than 1 outright. Verified
    # empirically that this combination is OpenAI-only — Anthropic's reasoning-locked models
    # (e.g. claude-opus-5) have no equivalent escape hatch and hard-require temperature=1, so
    # if `settings.llm_fallback_order` ever falls through to Anthropic for one of these calls,
    # it will fail instead of gracefully falling back. Acceptable today since OpenAI is the
    # primary provider.
    response = await router.complete(
        messages, response_format=ArchitectureQuestionListResult, temperature=0, reasoning_effort="none"
    )
    result = ArchitectureQuestionListResult.model_validate(
        load_json_response(response.choices[0].message.content)
    )

    if _problem_count(result, transcript_text) > 0:
        # temperature=0 makes a *good* run reliably reproducible, but it just as reliably
        # reproduces a *problematic* one — verified empirically: the same transcript gave
        # the exact same truncated result three runs in a row at temperature=0. Retrying at
        # the same temperature would almost certainly repeat it again, so each retry below
        # samples at a nonzero temperature instead, as an escape hatch from that determinism
        # — not a general retry-on-any-failure policy. Two independent triggers land here:
        # shallow (see `_looks_shallow`) and ungrounded (a `mentioned_components` name that
        # isn't actually in the transcript — see `_ungrounded_component_names`); either is
        # reason enough to retry. Stops as soon as a retry has zero problems; otherwise keeps
        # whichever candidate seen so far has fewer problems (`_problem_count`), tie-breaking
        # on more questions — not just the last attempt — and gives up after
        # SHALLOW_RETRY_ATTEMPTS rather than looping or spending indefinitely.
        for _ in range(SHALLOW_RETRY_ATTEMPTS):
            retry_response = await router.complete(
                messages,
                response_format=ArchitectureQuestionListResult,
                temperature=SHALLOW_RETRY_TEMPERATURE,
                reasoning_effort="none",
            )
            retry_result = ArchitectureQuestionListResult.model_validate(
                load_json_response(retry_response.choices[0].message.content)
            )
            retry_problems = _problem_count(retry_result, transcript_text)
            current_problems = _problem_count(result, transcript_text)
            if retry_problems < current_problems or (
                retry_problems == current_problems and len(retry_result.questions) > len(result.questions)
            ):
                result = retry_result
            if _problem_count(result, transcript_text) == 0:
                break

    _write_json_audit_file(ingestion_date_str, bronze_documents, "questions", result.model_dump_json(indent=2))

    return result


async def generate_data_contract_questions_for_batch(
    ingestion_date_str: str,
    bronze_documents: list[BronzeRow],
    mentioned_data_contracts: list[MentionedDataContractItem],
) -> DataContractQuestionListResult:
    """Question-generation stage 2 of 2 (`doc/silver_process.md` §3 node 3): one LLM call
    reads the pooled transcript against `prompts/data_contract_template.md`
    and drafts the full ODCS-completeness question set for *exactly* the contracts
    `mentioned_data_contracts` names — see `prompts/data_contract_questions.
    jinja`. Never discovers a contract on its own; an empty `mentioned_data_contracts` yields
    an empty result immediately, no LLM call spent on nothing to ask about.

    Same call shape as `generate_architecture_questions_for_batch` otherwise: no DB session,
    no LangGraph state, writes its own audit file
    (`output/ingestion_date=<date>/data_contract_questions/<transcription>.json`), same
    temperature/reasoning_effort/shallow-retry discipline, scoped to this stage's own
    shallowness/grounding checks (`_looks_shallow_data_contracts`/
    `_ungrounded_contract_targets`).
    """
    if not mentioned_data_contracts:
        result = DataContractQuestionListResult(questions=[])
        _write_json_audit_file(
            ingestion_date_str, bronze_documents, "data_contract_questions", result.model_dump_json(indent=2)
        )
        return result

    transcript_text = " ".join(row["content"] for row in bronze_documents)
    template_text = load_data_contract_template()
    messages = build_data_contract_question_generation_prompt(
        template_text, transcript_text, mentioned_data_contracts
    )
    response = await router.complete(
        messages, response_format=DataContractQuestionListResult, temperature=0, reasoning_effort="none"
    )
    result = DataContractQuestionListResult.model_validate(
        load_json_response(response.choices[0].message.content)
    )

    if _problem_count_data_contracts(result, mentioned_data_contracts) > 0:
        for _ in range(SHALLOW_RETRY_ATTEMPTS):
            retry_response = await router.complete(
                messages,
                response_format=DataContractQuestionListResult,
                temperature=SHALLOW_RETRY_TEMPERATURE,
                reasoning_effort="none",
            )
            retry_result = DataContractQuestionListResult.model_validate(
                load_json_response(retry_response.choices[0].message.content)
            )
            retry_problems = _problem_count_data_contracts(retry_result, mentioned_data_contracts)
            current_problems = _problem_count_data_contracts(result, mentioned_data_contracts)
            if retry_problems < current_problems or (
                retry_problems == current_problems and len(retry_result.questions) > len(result.questions)
            ):
                result = retry_result
            if _problem_count_data_contracts(result, mentioned_data_contracts) == 0:
                break

    _write_json_audit_file(
        ingestion_date_str, bronze_documents, "data_contract_questions", result.model_dump_json(indent=2)
    )

    return result
