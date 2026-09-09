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
from db.models import BronzeDocument
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


def _ungrounded_component_names(result: ArchitectureQuestionListResult, transcript: str) -> list[str]:
    """Every `mentioned_components` name that doesn't appear in the transcript verbatim
    (case-insensitive) — a hallucinated or paraphrased component name, not something the
    transcript actually said. Same deterministic check `testing_arch_questions_acb`'s
    golden-set test already runs on the side, moved into production so a real
    `make questions`/`make clarify` run gets the same escape-hatch retry the test does, not
    just a post-hoc failure report."""
    transcript_lower = transcript.lower()
    return [c.name for c in result.mentioned_components if c.name.lower() not in transcript_lower]


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


async def load_bronze_rows(ingestion_date_str: str, db: AsyncSession) -> list[BronzeRow]:
    """Every `bronze_documents` row for this `ingestion_date`, in insertion order.
    Raises `NoBronzeDocumentsError` if there are none — see that class's docstring."""
    ingestion_date = parse_ingestion_date(ingestion_date_str)
    result = await db.execute(
        select(BronzeDocument.source_component, BronzeDocument.content)
        .where(BronzeDocument.ingestion_date == ingestion_date)
        .order_by(BronzeDocument.id)
    )
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

    `architecture_diagram` is always empty in production today — there is no source yet for
    "the architecture as of this transcript's own point in time" (Gold doesn't version
    diagrams, `doc/silver_process.md` §7); the prompt's own Jinja `{% if known_architecture %}`
    already treats that as "no prior architecture known," a correct, expected case.
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
