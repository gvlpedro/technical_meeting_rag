"""The Silver clarification loop, from start to end. This turns `doc/silver_process.md` §3
into code.

This is one large module on purpose, instead of one file per node. The ten nodes below only
make sense when wired together. See `.tmp/tasks.md` Task 6 for why splitting this into
separate commits would mean landing broken states in between.
"""

import os

os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

import asyncio
import re
from pathlib import Path
from typing import Literal

import logfire
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from sqlalchemy import delete, select

from agents.shared import (
    SHALLOW_RETRY_ATTEMPTS,
    SHALLOW_RETRY_TEMPERATURE,
    distinct_sources,
    extract_authors_line,
    insert_authors_line,
    insert_source_line,
    latest_document_content,
    load_bronze_rows,
    mentions_grounded_in_source,
    source_content,
    transcription_base_name,
)
from agents.stages import gold
from agents.stages.adr_critic.prompts import build_critic_prompt
from agents.stages.adr_critic.schemas import CritiqueResult
from agents.stages.adr_generation.prompts import build_adr_generation_prompt
from agents.stages.adr_generation.service import (
    SUGGEST_INFO_CORRECTION_MESSAGE,
    adr_drops_suggest_info_content,
    adr_has_placeholder_leak,
    strip_diagram_colors,
)
from agents.stages.architecture_questions.service import (
    generate_architecture_questions_for_batch,
    previous_architecture_context,
)
from agents.stages.classification.prompts import build_classification_prompt
from agents.stages.classification.schemas import ClassificationResult, QuestionClassification
from agents.stages.data_contract_questions.service import generate_data_contract_questions_for_batch
from agents.stages.gold.schemas import ArchitecturePayload, ComponentPayload, DataContractPayload
from agents.state import ClarificationItem, SilverState
from agents.template import load_json_response
from app.config import settings
from db.models import SilverChunk, SilverClarification, SilverDocument
from db.session import async_session_factory
from ingestion.bronze_documents_chunker import parse_ingestion_date
from ingestion.embedder import embed
from llm import router

logfire.configure(
    token=settings.logfire_token,
    send_to_logfire="if-token-present",
    service_name="silver-clarification-loop",
)

_DECLINE_PHRASES = {"", "unknown", "n/a", "idk", "i don't know", "[irrelevant]","none"}
_DOWNGRADE_MARKER = " **[unknown — flagged by review]**"

# The frontend's per-question "Infer an answer" and "Suggest info" buttons.
# "Irrelevant" is different: it really does mean "no answer"
INFER_FROM_CONTEXT_MARKER = "[INFER FROM CONTEXT]"
SUGGEST_INFO_MARKER = "[SUGGEST INFO]"

_QUESTIONS_PRIORITIES = {
    "adr": 0,      # Decisions
    "component": 1,
    "architecture": 2,
    "change_impact": 3,
    "migration": 4,
    "metadata": 5,
}


def _top_questions(
    questions: list[str],
    architecture_cap: int,
    data_contract_cap: int,
    scopes: dict[str, str] | None = None,
) -> list[str]:
    """Deduplicates the questions then ranked by `_QUESTIONS_PRIORITIES`"""
    questions = list(dict.fromkeys(questions))
    if scopes is None:
        return questions[:architecture_cap]

    architecture_qs = [q for q in questions if scopes.get(q) != "data_contract"]
    data_contract_qs = [q for q in questions if scopes.get(q) == "data_contract"]
    architecture_qs.sort(key=lambda q: _QUESTIONS_PRIORITIES.get(scopes.get(q, ""), len(_QUESTIONS_PRIORITIES)))

    return architecture_qs[:architecture_cap] + data_contract_qs[:data_contract_cap]


def checkpointer_dsn() -> str:
    """Postgres reference"""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


def _is_inside_code_fence(content: str, index: int) -> bool:
    """Returns `True` if `index` falls inside a ` ``` `-fenced block"""
    return content.count("```", 0, index) % 2 == 1


def _downgrade_claim(content: str, claim: str) -> str:
    """Marks the first claim outside Mermaid fences, leaving Mermaid diagrams unchanged."""
    if not claim:
        return content
    marked = claim + _DOWNGRADE_MARKER
    if marked in content:
        return content  # Already downgraded. Running this again does nothing new.

    search_from = 0
    while True:
        index = content.find(claim, search_from)
        if index == -1:
            return content
        if not _is_inside_code_fence(content, index):
            return content[:index] + marked + content[index + len(claim) :]
        search_from = index + 1


async def load_bronze(state: SilverState) -> dict:
    async with async_session_factory() as session:
        rows = await load_bronze_rows(
            state["ingestion_date"],
            session,
            tenant=state["tenant"],
            source_components=state["source_components"],
        )

    return {
        "bronze_documents": rows,
        "transcript_text": " ".join(r["content"] for r in rows),
    }


async def generate_architecture_questions(state: SilverState) -> dict:
    """Generates architecture questions using the known current architecture and previous ADRs."""
    known_architecture_parts = []
    async with async_session_factory() as session:
        gold_diagram = strip_diagram_colors(await gold.current_architecture_diagram(session, tenant=state["tenant"]))
        if gold_diagram:
            known_architecture_parts.append(
                "### Tenant-wide current architecture (Gold, across every source)\n\n"
                f"```mermaid\n{gold_diagram}\n```"
            )
        for source in distinct_sources(state["bronze_documents"]):
            previous = await latest_document_content(session, state["tenant"], source)
            context = previous_architecture_context(previous)
            if context:
                known_architecture_parts.append(f"### {source}\n\n{context}")
    known_architecture = "\n\n".join(known_architecture_parts)

    result = await generate_architecture_questions_for_batch(
        state["ingestion_date"], state["bronze_documents"], architecture_diagram=known_architecture
    )
    # A name the LLM put in both lists is really a data contract wearing a component's
    # and a data contract is often named after the topic or queue it travels over.
    contract_names = {c.name.lower() for c in result.mentioned_data_contracts}
    components = [c for c in result.mentioned_components if c.name.lower() not in contract_names]
    return {
        "generated_questions": [q.model_dump() for q in result.questions],
        "mentioned_components": [c.model_dump() for c in components],
        "mentioned_data_contracts": [c.model_dump() for c in result.mentioned_data_contracts],
    }


def _slugify(name: str) -> str:
    """Lowercases the name and collapses runs of non-alphanumeric characters"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "contract"


def _ensure_schema_questions(questions: list[dict], mentioned_data_contracts: list[dict]) -> list[dict]:
    """This is a mechanical backstop for `prompts/data_contract_questions/questions.jinja`'s
    own FINAL SELF-CHECK"""
    targets_with_schema_question = {
        q["target"] for q in questions if q.get("scope") == "data_contract" and "schema" in q.get("id", "")
    }
    result = list(questions)
    for contract in mentioned_data_contracts:
        name = contract["name"]
        if name in targets_with_schema_question:
            continue
        result.append(
            {
                "id": f"contract.{_slugify(name)}.schema",
                "scope": "data_contract",
                "target": name,
                "requirement": "schema",
                "question": (
                    f"What fields (with type, required/nullable status, and meaning) make up "
                    f"the {name} data contract's schema?"
                ),
            }
        )
    return result


def _drop_new_contract_version_questions(questions: list[dict], mentioned_data_contracts: list[dict]) -> list[dict]:
    """This is a mechanical backstop for `prompts/data_contract_questions/questions.jinja`'s
    own PHASE 3 rule: a brand-new contract's version is always `1.0.0`"""
    new_contract_names = {c["name"] for c in mentioned_data_contracts if c.get("action") == "new"}
    return [
        q for q in questions if not (q.get("target") in new_contract_names and "version" in q.get("id", ""))
    ]


async def generate_data_contract_questions(state: SilverState) -> dict:
    """Generates data contract questions and mentions."""
    result = await generate_data_contract_questions_for_batch(
        state["ingestion_date"], state["bronze_documents"], state["mentioned_data_contracts"]
    )
    contract_questions = _ensure_schema_questions(
        [q.model_dump() for q in result.questions], state["mentioned_data_contracts"]
    )
    contract_questions = _drop_new_contract_version_questions(contract_questions, state["mentioned_data_contracts"])
    return {
        "generated_questions": state["generated_questions"] + contract_questions,
    }


def _canonical_classification(
    classification_id: str,
    by_classification_id: dict[str, QuestionClassification],
    seen: frozenset[str],
) -> QuestionClassification:
    """classify questions."""
    classification = by_classification_id[classification_id]
    target_id = classification.duplicate_of
    if target_id is None or target_id == classification_id or target_id in seen or target_id not in by_classification_id:
        return classification
    return _canonical_classification(target_id, by_classification_id, seen | {classification_id})


async def classify_questions(state: SilverState) -> dict:
    """This is the single decision that gates every question """
    questions = state["generated_questions"]
    messages = build_classification_prompt(questions, state["transcript_text"])
    response = await router.complete(
        messages, response_format=ClassificationResult, temperature=0, reasoning_effort="none"
    )
    result = ClassificationResult.model_validate(load_json_response(response.choices[0].message.content))

    by_id = {q["id"]: q for q in questions}
    by_classification_id = {c.id: c for c in result.classifications}
    clarifications: list[ClarificationItem] = []
    for c in result.classifications:
        question = by_id.get(c.id)
        if question is None:
            continue
        # A duplicate question is folded into the one it duplicates
        canonical = _canonical_classification(c.id, by_classification_id, frozenset())
        canonical_question = by_id.get(canonical.id, question)
        clarifications.append(
            {
                "id": question["id"],
                "scope": question["scope"],
                "target": question["target"],
                "requirement": question["requirement"],
                "question": canonical_question["question"],
                "answer": canonical.answer,
                "status": canonical.status,
            }
        )
    needs_clarification = [c for c in clarifications if c["status"] == "needs_clarification"]
    pending = _top_questions(
        [c["question"] for c in needs_clarification],
        state["max_architecture_pending_questions"],
        state["max_data_contract_pending_questions"],
        scopes={c["question"]: c["scope"] for c in needs_clarification},
    )

    return {
        "clarifications": clarifications,
        "pending_questions": pending,
        "interrupt_origin": "classify",
    }


def route_after_classify(state: SilverState) -> Literal["ask_human", "synthesize_document"]:
    return "ask_human" if state["pending_questions"] else "synthesize_document"


def ask_human(state: SilverState) -> dict:
    """This sends one `interrupt()` carrying every pending question together"""
    payload = {
        "ingestion_date": state["ingestion_date"],
        "pending_questions": state["pending_questions"],
        "origin": state["interrupt_origin"],
    }
    human_answers: dict[str, str] = interrupt(payload)

    updated = list(state["clarifications"])
    existing_questions = {item["question"] for item in updated}

    for question in state["pending_questions"]:
        raw_answer = human_answers.get(question, "")
        declined = raw_answer.strip().lower() in _DECLINE_PHRASES
        answer = None if declined else raw_answer.strip()
        status: Literal["unknown", "human_answered"] = "unknown" if declined else "human_answered"

        if question in existing_questions:
            for item in updated:
                if item["question"] == question:
                    item["answer"] = answer
                    item["status"] = status
        else:
            # This is a Boss-origin escalation question (see boss_decide)
            updated.append(
                {
                    "id": f"boss-escalation:{len(updated)}",
                    "scope": "change_impact",
                    "target": state["ingestion_date"],
                    "requirement": "contradiction_resolution",
                    "question": question,
                    "answer": answer,
                    "status": status,
                }
            )

    return {"clarifications": updated, "pending_questions": []}


async def synthesize_document(state: SilverState) -> dict:
    """ADR generator"""
    sources = state["redraft_only"] or distinct_sources(state["bronze_documents"])
    qa_pairs = [{"question": c["question"], "answer": c["answer"]} for c in state["clarifications"]]

    documents = dict(state["documents"])
    async with async_session_factory() as session:
        previous_diagram = strip_diagram_colors(
            await gold.current_architecture_diagram(session, tenant=state["tenant"])
        )
        for source in sources:
            messages = build_adr_generation_prompt(
                source_content(state["bronze_documents"], source),
                qa_pairs,
                previous_diagram,
                mentioned_components=state["mentioned_components"],
                mentioned_data_contracts=state["mentioned_data_contracts"],
            )
            response = await router.complete(messages, temperature=0, reasoning_effort="none")
            content = response.choices[0].message.content

            for _ in range(SHALLOW_RETRY_ATTEMPTS):
                dropped_suggestion = adr_drops_suggest_info_content(content, qa_pairs)
                if not adr_has_placeholder_leak(content) and not dropped_suggestion:
                    break
                retry_messages = (
                    [*messages, {"role": "assistant", "content": content}, {"role": "user", "content": SUGGEST_INFO_CORRECTION_MESSAGE}]
                    if dropped_suggestion
                    else messages
                )
                retry_response = await router.complete(
                    retry_messages, temperature=SHALLOW_RETRY_TEMPERATURE, reasoning_effort="none"
                )
                content = retry_response.choices[0].message.content
            documents[source] = content

    return {
        "documents": documents,
        "active_sources": sources,
        "redraft_only": None,
    }


async def critic_document(state: SilverState) -> dict:
    """Step to evaluate `completeness_score` and `unresolved_points`"""
    critiques = dict(state["critiques"])
    adr_scores = dict(state["adr_scores"])
    adr_unresolved_points = dict(state["adr_unresolved_points"])
    for source in state["active_sources"]:
        messages = build_critic_prompt(
            state["documents"][source],
            source_content(state["bronze_documents"], source),
            state["clarifications"],
        )
        response = await router.complete(messages, response_format=CritiqueResult)
        result = CritiqueResult.model_validate(load_json_response(response.choices[0].message.content))
        critiques[source] = [c.model_dump() for c in result.claims]
        adr_scores[source] = result.completeness_score
        adr_unresolved_points[source] = result.unresolved_points

    return {
        "critiques": critiques,
        "adr_scores": adr_scores,
        "adr_unresolved_points": adr_unresolved_points,
    }


def boss_decide(state: SilverState) -> dict:
    """No LLM call. Downgrades every low-severity claim from the Critic directly in the
    document; for a material claim, escalates to the human once — if it's still unresolved on
    retry, downgrades it too instead of asking again, so no source loops forever."""
    documents = dict(state["documents"])
    boss_verdicts = dict(state["boss_verdicts"])
    revision_attempted = dict(state["revision_attempted"])
    escalate_sources: list[str] = []
    pending_questions: list[str] = []

    for source in state["active_sources"]:
        claims = state["critiques"].get(source, [])
        material = [c for c in claims if c["severity"] == "material"]
        low = [c for c in claims if c["severity"] == "low"]

        content = documents[source]
        for claim in low:
            content = _downgrade_claim(content, claim["claim"])

        if not material:
            documents[source] = content
            boss_verdicts[source] = "ok"
            continue

        if not revision_attempted.get(source, False):
            documents[source] = content
            boss_verdicts[source] = "needs_human_review"
            revision_attempted[source] = True
            escalate_sources.append(source)
            for claim in material:
                pending_questions.append(
                    f'[{source}] The transcript does not support this claim: "{claim["claim"]}" '
                    f'— {claim["rationale"]}. What is actually correct?'
                )
        else:
            for claim in material:
                content = _downgrade_claim(content, claim["claim"])
            documents[source] = content
            boss_verdicts[source] = "ok"

    update: dict = {
        "documents": documents,
        "boss_verdicts": boss_verdicts,
        "revision_attempted": revision_attempted,
    }
    if escalate_sources:
        update["redraft_only"] = escalate_sources
        update["pending_questions"] = _top_questions(
            pending_questions,
            state["max_architecture_pending_questions"],
            state["max_data_contract_pending_questions"],
        )
        update["interrupt_origin"] = "boss"
    return update


def route_after_boss(state: SilverState) -> Literal["ask_human", "write_document"]:
    return "ask_human" if state["pending_questions"] else "write_document"


async def _persist_document_version(
    session,
    ingestion_date,
    source_component: str,
    content: str,
    mentioned_component_names: list[dict],
    mentioned_data_contract_names: list[dict],
    tenant: str,
) -> int:
    """Writes one `SilverDocument` row, deciding the version from the content

    `mentioned_component_names` and `mentioned_data_contract_names` get refreshed

    `authored_by` """
    new_hash = gold.content_hash(content)
    authored_by = extract_authors_line(content)
    latest = (
        await session.execute(
            select(SilverDocument)
            .where(SilverDocument.tenant == tenant, SilverDocument.source_component == source_component)
            .order_by(SilverDocument.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    def _row_fields(version: int) -> dict:
        return {
            "tenant": tenant,
            "ingestion_date": ingestion_date,
            "source_component": source_component,
            "version": version,
            "content": content,
            "content_hash": new_hash,
            "mentioned_component_names": mentioned_component_names,
            "mentioned_data_contract_names": mentioned_data_contract_names,
            "authored_by": authored_by,
        }

    if latest is None:
        version = 1
        session.add(SilverDocument(**_row_fields(version)))
    elif latest.content_hash == new_hash:
        version = latest.version
        for field, value in _row_fields(version).items():
            setattr(latest, field, value)
    else:
        version = latest.version + 1
        session.add(SilverDocument(**_row_fields(version)))
    return version


def _write_adr_audit_file(ingestion_date_str: str, source_component: str, content: str) -> None:
    """Writes `output/ingestion_date=<date>/adr/<transcription>.md`"""
    adr_dir = Path(settings.output_dir) / f"ingestion_date={ingestion_date_str}" / "adr"
    adr_dir.mkdir(parents=True, exist_ok=True)
    (adr_dir / f"{transcription_base_name(source_component)}.md").write_text(content, encoding="utf-8")


async def write_document(state: SilverState) -> dict:
    """Persists each synthesized ADR using its `(source_component, version)` pair and also
    writes an audit copy to disk. This only happens when `state["persist"]` is true.
    Drafts are not persisted until the user publishes them.

    Clarifications are always stored in `silver_clarifications` as a new row per
    (source, question), preserving the full Q&A history. This history is used when
    regenerating documents or asking additional questions.

    Mentioned components and data contracts are generated once for the whole batch.
    Before saving them, this function keeps only the mentions grounded in the current
    source, so Gold can use the resulting per-ADR lists directly.

    Finally, the function adds the author and source information to each document after
    the Critic and decision steps have finished. The stamped documents are returned even
    for drafts, so the frontend preview matches the eventually published ADR."""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    documents = {
        source: insert_source_line(insert_authors_line(content, state["username"]), source)
        for source, content in state["documents"].items()
    }
    document_versions: dict[str, int] = dict(state["document_versions"])
    async with async_session_factory() as session:
        for source, content in documents.items():
            if state["persist"]:
                source_text = source_content(state["bronze_documents"], source)
                document_versions[source] = await _persist_document_version(
                    session,
                    ingestion_date,
                    source,
                    content,
                    mentions_grounded_in_source(source_text, state["mentioned_components"]),
                    mentions_grounded_in_source(source_text, state["mentioned_data_contracts"]),
                    state["tenant"],
                )
                _write_adr_audit_file(state["ingestion_date"], source, content)

            for item in state["clarifications"]:
                session.add(
                    SilverClarification(
                        tenant=state["tenant"],
                        ingestion_date=ingestion_date,
                        source_component=source,
                        question=item["question"],
                        answer=item["answer"],
                    )
                )
        await session.commit()

    return {"documents": documents, "document_versions": document_versions}


def route_after_write_document(state: SilverState) -> Literal["chunk_and_embed", "skip_to_end"]:
    """A frontend draft ends right here"""
    return "chunk_and_embed" if state["persist"] else "skip_to_end"


async def chunk_and_embed(state: SilverState) -> dict:
    """This makes one embedding per ADR, with no token-splitting: an ADR is retrieved as a
    whole, never as a fragment"""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    async with async_session_factory() as session:
        for source, content in state["documents"].items():
            if not content.strip():
                continue
            version = state["document_versions"][source]
            await session.execute(
                delete(SilverChunk).where(
                    SilverChunk.tenant == state["tenant"],
                    SilverChunk.source_component == source,
                    SilverChunk.version == version,
                )
            )
            [embedding] = await asyncio.to_thread(embed, [content])
            session.add(
                SilverChunk(
                    tenant=state["tenant"],
                    ingestion_date=ingestion_date,
                    source_component=source,
                    version=version,
                    content=content,
                    embedding=embedding,
                )
            )
        await session.commit()
    return {}


async def extract_gold_facts(state: SilverState) -> dict:
    """This makes one structured-extraction LLM call per source that meets two conditions:
    it has a document written this run (`state["documents"]` and its `boss_verdicts` value is `"ok"`"""
    gold_extractions = dict(state["gold_extractions"])
    async with async_session_factory() as session:
        for source in state["documents"]:
            if state["boss_verdicts"].get(source) != "ok":
                continue
            version = state["document_versions"][source]
            if await gold.already_extracted(session, source, version, tenant=state["tenant"]):
                continue

            result = await gold.extract_gold_facts_for_source(state["documents"][source])
            gold_extractions[source] = result.model_dump()

    return {"gold_extractions": gold_extractions}


async def resolve_gold_identity(state: SilverState) -> dict:
    """No LLM call happens here. This tries an exact match on `gold_aliases.alias` first,
    then a `pg_trgm` fuzzy match, and finally mints a new `entity_id` (`gold.resolve_entity_id`)
    if neither match works. It does this for every name this pass's extractions touch"""
    entity_id_maps = dict(state["gold_entity_ids"])

    async with async_session_factory() as session:
        for source, extraction in state["gold_extractions"].items():
            version = state["document_versions"][source]
            name_to_id: dict[str, str] = dict(entity_id_maps.get(source, {}))

            unknown_component_names = {c["name"] for c in extraction["components"] if c["status"] == "unknown"}
            unknown_contract_names = {c["name"] for c in extraction["contracts"] if c["action"] == "unknown"}

            for component in extraction["components"]:
                if component["status"] == "unknown":
                    continue
                await gold.resolve_and_alias(
                    session, "component", component["name"], name_to_id, source, version, tenant=state["tenant"]
                )
                for dep_name in component.get("dependency_names", []):
                    if dep_name in unknown_component_names:
                        continue
                    await gold.resolve_and_alias(
                        session, "component", dep_name, name_to_id, source, version, tenant=state["tenant"]
                    )
                for contract_name in component.get("contract_names", []):
                    if contract_name in unknown_contract_names:
                        continue
                    await gold.resolve_and_alias(
                        session, "data_contract", contract_name, name_to_id, source, version, tenant=state["tenant"]
                    )

            for contract in extraction["contracts"]:
                if contract["action"] == "unknown":
                    continue
                await gold.resolve_and_alias(
                    session, "data_contract", contract["name"], name_to_id, source, version, tenant=state["tenant"]
                )
                for component_name in (contract["producer"], contract["consumer"]):
                    if component_name in unknown_component_names:
                        continue
                    await gold.resolve_and_alias(
                        session, "component", component_name, name_to_id, source, version, tenant=state["tenant"]
                    )

            entity_id_maps[source] = name_to_id
        await session.commit()

    return {"gold_entity_ids": entity_id_maps}


async def _persist_components(
    session, extraction: dict, name_to_id: dict, source: str, version: int, ingestion_date, tenant: str,
    authored_by: str,
) -> None:
    for component in extraction["components"]:
        if component["status"] == "unknown":
            continue

        payload = ComponentPayload(
            dependency_ids=sorted(
                {name_to_id[n] for n in component.get("dependency_names", []) if n in name_to_id}
            ),
            contract_ids=sorted(
                {name_to_id[n] for n in component.get("contract_names", []) if n in name_to_id}
            ),
        ).model_dump()
        await gold.persist_entity_version(
            session,
            entity_type="component",
            entity_id=name_to_id[component["name"]],
            canonical_name=component["name"],
            operation=component["status"],
            narrative=component["narrative"],
            payload=payload,
            source_component=source,
            source_adr_version=version,
            ingestion_date=ingestion_date,
            tenant=tenant,
            authored_by=authored_by,
        )


async def _persist_contracts(
    session, extraction: dict, name_to_id: dict, source: str, version: int, ingestion_date, tenant: str,
    authored_by: str,
) -> None:
    for contract in extraction["contracts"]:
        if contract["action"] == "unknown":
            continue
        payload = DataContractPayload(
            producer=contract["producer"],
            consumer=contract["consumer"],
            producer_id=name_to_id.get(contract["producer"], ""),
            consumer_id=name_to_id.get(contract["consumer"], ""),
            odcs_spec=gold.parse_odcs_spec(contract.get("odcs_spec", "")),
        ).model_dump()
        await gold.persist_entity_version(
            session,
            entity_type="data_contract",
            entity_id=name_to_id[contract["name"]],
            canonical_name=contract["name"],
            operation=contract["action"],
            narrative=contract["narrative"],
            payload=payload,
            source_component=source,
            source_adr_version=version,
            ingestion_date=ingestion_date,
            tenant=tenant,
            authored_by=authored_by,
        )


async def _persist_architecture(
    session, extraction: dict, source: str, version: int, ingestion_date, tenant: str, authored_by: str
) -> None:
    """This is scoped per source_component"""
    # These are sorted for the same reason as ComponentPayload's dependency_ids and
    # contract_ids above: order must not affect `_entity_hash`
    payload = ArchitecturePayload(
        mermaid_diagram=extraction.get("mermaid_diagram", ""),
        components=sorted({c["name"] for c in extraction["components"]}),
        dependencies=sorted({dep for c in extraction["components"] for dep in c.get("dependency_names", [])}),
    ).model_dump()
    await gold.persist_entity_version(
        session,
        entity_type="architecture",
        entity_id=f"architecture:{source}",
        canonical_name=source,
        operation=extraction["architecture_change"],
        narrative=extraction["architecture_narrative"],
        payload=payload,
        source_component=source,
        source_adr_version=version,
        ingestion_date=ingestion_date,
        tenant=tenant,
        authored_by=authored_by,
    )


async def persist_gold_evolution(state: SilverState) -> dict:
    """No LLM call happens here. This compares hashes and bumps a version, per entity"""
    ingestion_date = parse_ingestion_date(state["ingestion_date"])
    async with async_session_factory() as session:
        for source, extraction in state["gold_extractions"].items():
            version = state["document_versions"][source]
            name_to_id = state["gold_entity_ids"].get(source, {})
            tenant = state["tenant"]
            authored_by = extract_authors_line(state["documents"][source])
            await _persist_components(
                session, extraction, name_to_id, source, version, ingestion_date, tenant, authored_by
            )
            await _persist_contracts(
                session, extraction, name_to_id, source, version, ingestion_date, tenant, authored_by
            )
            await _persist_architecture(session, extraction, source, version, ingestion_date, tenant, authored_by)
        await session.commit()

    return {}

def _add_modes(graph: StateGraph) -> None:
    graph.add_node("load_bronze", load_bronze)
    graph.add_node("generate_architecture_questions", generate_architecture_questions)
    graph.add_node("generate_data_contract_questions", generate_data_contract_questions)
    graph.add_node("classify_questions", classify_questions)
    graph.add_node("ask_human", ask_human)
    graph.add_node("synthesize_document", synthesize_document) #TODO: change to generate_adr
    graph.add_node("critic_document", critic_document)
    graph.add_node("boss_decide", boss_decide)
    graph.add_node("write_document", write_document) #TODO: change to write_silver_document
    graph.add_node("chunk_and_embed", chunk_and_embed) #TODO: change to chunk_and_embed_silver_document
    graph.add_node("extract_gold_facts", extract_gold_facts)
    graph.add_node("resolve_gold_identity", resolve_gold_identity)
    graph.add_node("persist_gold_evolution", persist_gold_evolution)

def _add_edgestates(graph: StateGraph) -> None:
    graph.add_edge(START, "load_bronze")
    graph.add_edge("load_bronze", "generate_architecture_questions")
    graph.add_edge("generate_architecture_questions", "generate_data_contract_questions")
    graph.add_edge("generate_data_contract_questions", "classify_questions")
    graph.add_conditional_edges(
        "classify_questions",
        route_after_classify,
        {"ask_human": "ask_human", "synthesize_document": "synthesize_document"},
    )
    graph.add_edge("ask_human", "synthesize_document")
    graph.add_edge("synthesize_document", "critic_document")
    graph.add_edge("critic_document", "boss_decide")
    graph.add_conditional_edges(
        "boss_decide",
        route_after_boss,
        {"ask_human": "ask_human", "write_document": "write_document"},
    )
    graph.add_conditional_edges(
        "write_document",
        route_after_write_document,
        {"chunk_and_embed": "chunk_and_embed", "skip_to_end": END},
    )
    graph.add_edge("chunk_and_embed", "extract_gold_facts")
    graph.add_edge("extract_gold_facts", "resolve_gold_identity")
    graph.add_edge("resolve_gold_identity", "persist_gold_evolution")
    graph.add_edge("persist_gold_evolution", END)


def build_graph(checkpointer) -> CompiledStateGraph:
    graph = StateGraph(SilverState)
    _add_modes(graph)
    _add_edgestates(graph)
    return graph.compile(checkpointer=checkpointer)
