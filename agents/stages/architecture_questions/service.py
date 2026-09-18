"""Callable, independently-testable logic for the architecture-questions stage. This is
question-generation stage 1 of 2 in the pipeline. One LLM call reads the pooled transcript and
drafts a fresh, transcript-specific question list. It covers every component and every
architecture or ADR-level gap that the transcript raises but leaves incomplete. It also
identifies, but never fully specifies, every data contract in play. `agents.graph`'s
`generate_architecture_questions` node is a thin wrapper that calls
`generate_architecture_questions_for_batch` below. `make questions` and this module's own
tests and golden set (`agents/stages/architecture_questions/testing/`) call it directly.

We split question generation into two sequential stages, this one and then `agents.stages.
data_contract_questions`, instead of using one combined pass. This fixed a real failure mode
we observed: a single combined pass regularly under-covered data contracts, even when
component and architecture coverage was thorough."""

from agents.shared import (
    SHALLOW_RETRY_ATTEMPTS,
    SHALLOW_RETRY_TEMPERATURE,
    _markdown_section,
    _write_json_audit_file,
    name_appears_in_text,
)
from agents.stages.architecture_questions.prompts import (
    build_architecture_identification_prompt,
    build_architecture_question_drafting_prompt,
    build_architecture_question_generation_prompt,
    build_architecture_question_selection_prompt,
    load_architecture_template,
)
from agents.stages.architecture_questions.schemas import (
    ArchitectureIdentificationResult,
    ArchitectureQuestionListResult,
    DraftedQuestionsResult,
    SelectedQuestionsResult,
)
from agents.state import BronzeRow, GeneratedQuestion, MentionedComponentItem, MentionedDataContractItem
from agents.template import load_json_response
from llm import router


def _looks_shallow(result: ArchitectureQuestionListResult) -> bool:
    """This is a heuristic, not a hard rule. It catches a specific failure mode we have seen
    in practice. The Actor correctly identifies every component, with the right names and the
    right lifecycle status, but only drafts the most obvious per-component questions. It then
    stops before ever reaching architecture- or ADR-level questions for the same transcript.
    Either signal is enough to suspect this: too few questions for how many components were
    identified, or every single question sharing one scope even though more than one
    component is in play.
    """
    if not result.mentioned_components:
        return False  # There is nothing to judge shallowness against.
    if len(result.questions) < 2 * len(result.mentioned_components):
        return True
    scopes = {q.scope for q in result.questions}
    return len(scopes) == 1 and len(result.mentioned_components) > 1


def _ungrounded_component_names(result: ArchitectureQuestionListResult, transcript: str) -> list[str]:
    """Every `mentioned_components` name that does not appear in the transcript word for word.
    The check is case-insensitive and word-boundary-safe. See
    `agents.shared.name_appears_in_text`. A name that fails this check is a hallucinated or
    paraphrased component name, not something the transcript actually said. This is the same
    reliable check that `agents.stages.architecture_questions.testing`'s golden-set test already runs on the
    side. We moved it into production so a real `make questions` or `make clarify` run gets
    the same retry the test gets, instead of only a failure report after the fact."""
    transcript_lower = transcript.lower()
    return [c.name for c in result.mentioned_components if not name_appears_in_text(c.name, transcript_lower)]


def _problem_count(result: ArchitectureQuestionListResult, transcript: str) -> int:
    """Returns 0, 1, or 2: how many of the two retry triggers, shallow and ungrounded, a
    result has. We use this to rank candidates across retry attempts. Fewer problems always
    wins. Among candidates with an equal number of problems, including zero, more questions
    wins as a tie-break."""
    return int(_looks_shallow(result)) + int(bool(_ungrounded_component_names(result, transcript)))


async def generate_architecture_questions_for_batch(
    ingestion_date_str: str, bronze_documents: list[BronzeRow], architecture_diagram: str = ""
) -> ArchitectureQuestionListResult:
    """Question-generation stage 1 of 2 (`doc/silver_process.md` §3 node 2). One LLM call
    reads the pooled transcript against `prompts/architecture_questions/template.md`. It
    drafts a fresh, transcript-specific question list. The list covers every component and
    every architecture or ADR-level gap that the transcript raises but leaves incomplete. It
    also identifies, but never fully specifies, every data contract in play. See
    `prompts/architecture_questions/combined.jinja`. It also returns `mentioned_components`,
    each with its own lifecycle `status`, and `mentioned_data_contracts`, with name, producer,
    consumer, and action. `mentioned_data_contracts` feeds directly into
    `agents.stages.data_contract_questions.service.generate_data_contract_questions_for_batch`.

    This function uses no DB session and no LangGraph state. It takes `bronze_documents` in
    and returns the parsed result. As a side effect, it writes the audit file
    `output/ingestion_date=<date>/questions/<transcription>.json`. You can call it directly
    from a script or a unit test with `bronze_documents` built by hand. `db/session.py` never
    enters the picture.

    `architecture_diagram` is empty only for a source with no prior ADR at all.
    `agents.graph.generate_architecture_questions` passes
    `agents.stages.architecture_questions.service.previous_architecture_context`'s output
    here. That output is built from the previous `SilverDocument` for the same source, never
    from Gold. See that function's own docstring. The prompt's own Jinja
    `{% if known_architecture %}` already treats an empty value as "no prior architecture
    known." That is the correct, expected case for a source's first-ever run.
    """
    transcript_text = " ".join(row["content"] for row in bronze_documents)
    template_text = load_architecture_template()
    messages = build_architecture_question_generation_prompt(
        template_text, transcript_text, architecture_diagram=architecture_diagram
    )
    # temperature=0: this prompt is a mandatory, systematic checklist (see its own FINAL
    # SELF-CHECK section), not a creative-writing task. With the default sampling temperature,
    # the same transcript would sometimes get full coverage and sometimes skip parts of it
    # entirely, run to run, for no reason tied to the transcript itself. reasoning_effort=
    # "none" must go alongside it for a reasoning-locked model, such as gpt-5.6-sol/terra.
    # Those models otherwise reject any temperature other than 1 outright. We confirmed by
    # testing that this combination only works for OpenAI. Anthropic's reasoning-locked
    # models, such as claude-opus-5, have no equivalent option and always require
    # temperature=1. So if `settings.llm_fallback_order` ever falls through to Anthropic for
    # one of these calls, that call will fail instead of gracefully falling back. This is
    # acceptable today, since OpenAI is the primary provider.
    response = await router.complete(
        messages, response_format=ArchitectureQuestionListResult, temperature=0, reasoning_effort="none"
    )
    result = ArchitectureQuestionListResult.model_validate(
        load_json_response(response.choices[0].message.content)
    )

    if _problem_count(result, transcript_text) > 0:
        # temperature=0 makes a good run reliably reproducible, but it just as reliably
        # reproduces a problematic one. We confirmed this by testing: the same transcript gave
        # the exact same truncated result three runs in a row at temperature=0. Retrying at
        # the same temperature would almost certainly repeat the same problem. So each retry
        # below samples at a nonzero temperature instead, purely to break out of that
        # determinism. This is not a general retry-on-any-failure policy. Two independent
        # triggers land here: shallow (see `_looks_shallow`) and ungrounded, meaning a
        # `mentioned_components` name that is not actually in the transcript (see
        # `_ungrounded_component_names`). Either one is reason enough to retry. We stop as
        # soon as a retry has zero problems. Otherwise we keep whichever candidate seen so far
        # has fewer problems (`_problem_count`), tie-breaking on more questions, not just
        # keeping the last attempt. We give up after SHALLOW_RETRY_ATTEMPTS, rather than
        # looping or spending indefinitely.
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


def previous_architecture_context(previous_adr_content: str | None) -> str:
    """The previous ADR's Target Architecture diagram plus its Affected Components table,
    combined. This is this run's own `KNOWN_ARCHITECTURE` input for
    `generate_architecture_questions_for_batch`. It gives the LLM both the diagram, which
    shows relationships, and the named components with their last confirmed status. This lets
    the LLM classify a component as `new` or `unchanged` against something real, instead of
    guessing with no anchor at all (`prompts/architecture_questions/combined.jinja`'s own
    STATUS RULES). Returns `""` if there is no previous ADR for this source, that is, on a
    first-time run."""
    if not previous_adr_content:
        return ""
    return _markdown_section(
        previous_adr_content, "## 3. Target Architecture", "## 5. Affected Data Contracts"
    )


# --- Decomposed architecture-questions pipeline (evaluation prototype) ------------------------
#
# Three separate LLM calls do what `generate_architecture_questions_for_batch` above does in
# one call: identify, then draft broadly, then select the small, high-value set. This is not
# wired into `agents/graph.py`. That combined function is still what production actually
# calls. This pipeline exists so we can compare the two approaches on the same golden set
# (`agents/stages/architecture_questions/testing/`), before deciding whether the split is worth the extra LLM
# calls it costs. See TESTING_REPORT.md for that comparison once it exists. Each function
# below is directly callable and independently testable, the same philosophy as every other
# function in this module: no graph state, no DB session, just LLM calls in and a validated
# Pydantic result out.


async def identify_architecture_entities(
    transcript_text: str, architecture_diagram: str = ""
) -> ArchitectureIdentificationResult:
    """Call 1 of 3. It identifies components and data contracts only, and does not draft
    questions. It follows the same temperature=0 discipline as every other question-generation
    call in this module. See `agents.graph.classify_questions`'s docstring for why it is worth
    avoiding a non-zero temperature on a decision this important."""
    messages = build_architecture_identification_prompt(transcript_text, architecture_diagram)
    response = await router.complete(
        messages, response_format=ArchitectureIdentificationResult, temperature=0, reasoning_effort="none"
    )
    return ArchitectureIdentificationResult.model_validate(load_json_response(response.choices[0].message.content))


async def draft_architecture_questions(
    template_text: str,
    transcript_text: str,
    architecture_diagram: str,
    mentioned_components: list[MentionedComponentItem],
    mentioned_data_contracts: list[MentionedDataContractItem],
) -> DraftedQuestionsResult:
    """Call 2 of 3. It builds a broad, unfiltered candidate list from call 1's identification
    result. Unlike the combined function above, this call deliberately has no shallow-retry or
    problem-count logic. This call is meant to over-generate, so there is no "too few
    questions" failure mode to retry against. That judgment belongs entirely to call 3."""
    messages = build_architecture_question_drafting_prompt(
        template_text, transcript_text, architecture_diagram, mentioned_components, mentioned_data_contracts
    )
    response = await router.complete(
        messages, response_format=DraftedQuestionsResult, temperature=0, reasoning_effort="none"
    )
    return DraftedQuestionsResult.model_validate(load_json_response(response.choices[0].message.content))


async def select_architecture_questions(
    transcript_text: str, candidate_questions: list[GeneratedQuestion]
) -> SelectedQuestionsResult:
    """Call 3 of 3. It takes call 2's candidates and filters, deduplicates, and value-scores
    them down to the final set."""
    messages = build_architecture_question_selection_prompt(transcript_text, candidate_questions)
    response = await router.complete(
        messages, response_format=SelectedQuestionsResult, temperature=0, reasoning_effort="none"
    )
    return SelectedQuestionsResult.model_validate(load_json_response(response.choices[0].message.content))
