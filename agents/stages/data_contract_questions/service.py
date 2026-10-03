"""Logic for the data-contract-questions stage (question-generation stage 2 of 2). One LLM
call per batch drafts ODCS-completeness questions for the contracts stage 1
(`agents.stages.architecture_questions`) identified. It never discovers a contract on its own.
`agents.graph.generate_data_contract_questions` wraps `generate_data_contract_questions_for_batch`
below."""

from agents.shared import SHALLOW_RETRY_ATTEMPTS, SHALLOW_RETRY_TEMPERATURE, _write_json_audit_file
from agents.stages.data_contract_questions.prompts import (
    build_data_contract_question_generation_prompt,
    load_data_contract_template,
)
from agents.stages.data_contract_questions.schemas import DataContractQuestionListResult
from agents.state import BronzeRow, MentionedDataContractItem
from agents.template import load_json_response
from llm import router

# Floor for `_looks_shallow_data_contracts`. ODCS completeness spans several categories per
# contract, so a healthy run drafts well more than one question per contract.
MIN_QUESTIONS_PER_CONTRACT = 5


def _looks_shallow_data_contracts(
    result: DataContractQuestionListResult, mentioned_data_contracts: list[MentionedDataContractItem]
) -> bool:
    """Same check as `architecture_questions.service._looks_shallow`. Too few questions for
    the number of contracts given means some were skipped or only covered on the surface."""
    if not mentioned_data_contracts:
        return False
    return len(result.questions) < MIN_QUESTIONS_PER_CONTRACT * len(mentioned_data_contracts)


def _ungrounded_contract_targets(
    result: DataContractQuestionListResult, mentioned_data_contracts: list[MentionedDataContractItem]
) -> list[str]:
    """Every question's `target` must match a given contract (questions.jinja, PHASE 12 — NO
    INVENTION). A mismatch is an invented contract, or a component name in a contract slot."""
    known = {c["name"].lower() for c in mentioned_data_contracts}
    return sorted({q.target for q in result.questions if q.target.lower() not in known})


def _problem_count_data_contracts(
    result: DataContractQuestionListResult, mentioned_data_contracts: list[MentionedDataContractItem]
) -> int:
    return int(_looks_shallow_data_contracts(result, mentioned_data_contracts)) + int(
        bool(_ungrounded_contract_targets(result, mentioned_data_contracts))
    )


async def generate_data_contract_questions_for_batch(
    ingestion_date_str: str,
    bronze_documents: list[BronzeRow],
    mentioned_data_contracts: list[MentionedDataContractItem],
) -> DataContractQuestionListResult:
    """Question-generation stage 2 of 2 (`doc/silver_process.md` §3 node 3). One LLM call
    drafts ODCS-completeness questions for exactly the contracts `mentioned_data_contracts`
    names; it never discovers a contract on its own. An empty list skips the LLM call.

    Same call shape as `architecture_questions.service.generate_architecture_questions_for_batch`:
    no DB session, no LangGraph state, writes its own audit file, and retries on a shallow or
    ungrounded result."""
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
