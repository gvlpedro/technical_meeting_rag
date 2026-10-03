"""Logic for the architecture-questions stage (question-generation stage 1 of 2). One LLM call
drafts a question list covering every component and architecture/ADR-level gap, and identifies
(but does not fully specify) every data contract in play. `agents.graph.
generate_architecture_questions` wraps `generate_architecture_questions_for_batch` below.

Question generation splits into two stages — this one, then `data_contract_questions` — because
one combined pass under-covered data contracts even with thorough component coverage."""

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
    """A heuristic for a known failure: components identified correctly, but only the most
    obvious per-component questions drafted, with no architecture/ADR-level questions. Signals:
    too few questions for the components found, or every question sharing one scope with more
    than one component in play."""
    if not result.mentioned_components:
        return False  # Nothing to judge shallowness against.
    if len(result.questions) < 2 * len(result.mentioned_components):
        return True
    scopes = {q.scope for q in result.questions}
    return len(scopes) == 1 and len(result.mentioned_components) > 1


def _ungrounded_component_names(result: ArchitectureQuestionListResult, transcript: str) -> list[str]:
    """Every `mentioned_components` name missing from the transcript (case-insensitive,
    word-boundary-safe — see `agents.shared.name_appears_in_text`). A name that fails is
    hallucinated or paraphrased, not something the transcript actually said."""
    transcript_lower = transcript.lower()
    return [c.name for c in result.mentioned_components if not name_appears_in_text(c.name, transcript_lower)]


def _problem_count(result: ArchitectureQuestionListResult, transcript: str) -> int:
    """0, 1, or 2: how many retry triggers (shallow, ungrounded) a result has. Ranks retry
    candidates — fewer problems wins; more questions breaks a tie."""
    return int(_looks_shallow(result)) + int(bool(_ungrounded_component_names(result, transcript)))


async def generate_architecture_questions_for_batch(
    ingestion_date_str: str, bronze_documents: list[BronzeRow], architecture_diagram: str = ""
) -> ArchitectureQuestionListResult:
    """Question-generation stage 1 of 2 (`doc/silver_process.md` §3 node 2). One LLM call
    drafts a question list covering every component and architecture/ADR-level gap, and
    identifies (not fully) every data contract in play. Returns `mentioned_components` (each
    with lifecycle `status`) and `mentioned_data_contracts` (name, producer, consumer, action);
    the latter feeds `data_contract_questions.service.generate_data_contract_questions_for_batch`.

    Uses no DB session or LangGraph state — call it directly with hand-built
    `bronze_documents`. Writes an audit file as a side effect.

    `architecture_diagram` is empty only for a source with no prior ADR. It comes from
    `previous_architecture_context`'s output, built from the previous `SilverDocument`, never
    from Gold."""
    transcript_text = " ".join(row["content"] for row in bronze_documents)
    template_text = load_architecture_template()
    messages = build_architecture_question_generation_prompt(
        template_text, transcript_text, architecture_diagram=architecture_diagram
    )
    # temperature=0: this is a systematic checklist, not creative writing — a higher
    # temperature skips parts of the transcript inconsistently, run to run.
    # reasoning_effort="none" is required alongside it for a reasoning-locked OpenAI model.
    # Anthropic's reasoning-locked models have no such option and need temperature=1, so this
    # call fails if settings.llm_fallback_order falls through to Anthropic. Accepted for now,
    # since OpenAI is the primary provider.
    response = await router.complete(
        messages, response_format=ArchitectureQuestionListResult, temperature=0, reasoning_effort="none"
    )
    result = ArchitectureQuestionListResult.model_validate(
        load_json_response(response.choices[0].message.content)
    )

    if _problem_count(result, transcript_text) > 0:
        # temperature=0 reproduces a problematic run just as reliably as a good one, so each
        # retry samples at a nonzero temperature to break the determinism. Triggers: shallow
        # (`_looks_shallow`) or ungrounded (`_ungrounded_component_names`). Stops at zero
        # problems; otherwise keeps the candidate with fewer problems, more questions as a
        # tie-break; gives up after SHALLOW_RETRY_ATTEMPTS.
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
    """The previous ADR's Target Architecture diagram plus Affected Components table — this
    run's `KNOWN_ARCHITECTURE` input, so the LLM classifies `new`/`unchanged` against a real
    anchor (combined.jinja, STATUS RULES). Returns `""` on a first-time run."""
    if not previous_adr_content:
        return ""
    return _markdown_section(
        previous_adr_content, "## 3. Target Architecture", "## 5. Affected Data Contracts"
    )


# --- Decomposed architecture-questions pipeline (evaluation prototype) ------------------------
#
# Three LLM calls instead of one: identify, draft broadly, then select. Not wired into
# agents/graph.py — exists to compare against the combined function on the same golden set.
# Each function below is directly callable: no graph state, no DB session.


async def identify_architecture_entities(
    transcript_text: str, architecture_diagram: str = ""
) -> ArchitectureIdentificationResult:
    """Call 1 of 3: identifies components and data contracts only, drafts no questions.
    temperature=0, same as every other call in this module."""
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
    """Call 2 of 3: builds a broad, unfiltered candidate list from call 1's result. No
    shallow-retry logic — this call is meant to over-generate. Call 3 judges the result."""
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
    """Call 3 of 3: filters, deduplicates, and value-scores call 2's candidates down to the
    final set."""
    messages = build_architecture_question_selection_prompt(transcript_text, candidate_questions)
    response = await router.complete(
        messages, response_format=SelectedQuestionsResult, temperature=0, reasoning_effort="none"
    )
    return SelectedQuestionsResult.model_validate(load_json_response(response.choices[0].message.content))
