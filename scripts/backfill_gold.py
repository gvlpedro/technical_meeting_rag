#!/usr/bin/env python3
"""Replay Gold extraction/reconciliation over existing `silver_documents` rows, independent of
Silver's own graph run — `.tmp/gold_process_v5.md` §4's "fully rebuildable" promise. The three
Gold nodes in `agents/graph.py` (`extract_gold_facts`/`resolve_gold_identity`/
`persist_gold_evolution`) call plain functions in `agents/gold_service.py`, not graph-only
logic — this script calls those same functions directly instead of re-running Silver's ACB
loop, which would re-spend clarification/critic LLM calls and risk spurious human-in-the-loop
interrupts for transcripts that need no re-clarification at all. Use this after changing the
extraction prompt/model and wanting to reprocess history, or to backfill Gold for
`silver_documents` rows that predate Gold's existence.

Usage:
    uv run python3 scripts/backfill_gold.py                          # every silver_documents row
    uv run python3 scripts/backfill_gold.py --source-component meeting.en.vtt
    uv run python3 scripts/backfill_gold.py --force                  # re-extract even if
                                                                       # already_extracted is
                                                                       # true — use after
                                                                       # changing the extraction
                                                                       # prompt itself

KNOWN DUPLICATION (not fixed here, flagged for follow-up): identity resolution itself is shared
(`gold_service.resolve_and_alias`, used by both this script and `agents.graph.
resolve_gold_identity`), but the persist orchestration below (looping components/contracts,
building `ComponentPayload`/`DataContractPayload`/`ArchitecturePayload`) is still the same shape
as `agents.graph.persist_gold_evolution`, duplicated here rather than factored into one shared
`agents.gold_service` function both call. Left duplicated for this pass because the graph keeps
identity-resolution and persistence as two separate, separately-traced nodes on purpose (Logfire
span-per-concern, matching every other node in that graph) — collapsing them into one shared
function would mean either changing that node structure or having this script call two
functions that each reopen a loop over the same extraction result. A real fix factors the
shared persist-loop body into `agents/gold_service.py` once there's a second real caller
pressuring it to.
"""

import argparse
import asyncio

from sqlalchemy import select

from agents import gold_service
from agents.schemas import ArchitecturePayload, ComponentPayload, DataContractPayload
from db.models import SilverDocument
from db.session import async_session_factory


async def _backfill_one(session, doc: SilverDocument, force: bool) -> bool:
    if not force and await gold_service.already_extracted(session, doc.source_component, doc.version):
        return False

    result = await gold_service.extract_gold_facts_for_source(
        doc.content, doc.mentioned_component_names, doc.mentioned_data_contract_names
    )
    name_to_id: dict[str, str] = {}

    for component in result.components:
        if component.status == "unknown":
            continue
        await gold_service.resolve_and_alias(
            session, "component", component.name, name_to_id, doc.source_component, doc.version
        )
        for dep in component.dependency_names:
            await gold_service.resolve_and_alias(
                session, "component", dep, name_to_id, doc.source_component, doc.version
            )
        for contract_name in component.contract_names:
            await gold_service.resolve_and_alias(
                session, "data_contract", contract_name, name_to_id, doc.source_component, doc.version
            )
    for contract in result.contracts:
        if contract.action == "unknown":
            continue
        await gold_service.resolve_and_alias(
            session, "data_contract", contract.name, name_to_id, doc.source_component, doc.version
        )

    for component in result.components:
        if component.status == "unknown":
            continue
        # Sorted (deduplicated too) so list order never affects `_entity_hash` — see the
        # matching comment in `agents.graph.persist_gold_evolution`.
        payload = ComponentPayload(
            dependency_ids=sorted({name_to_id[n] for n in component.dependency_names if n in name_to_id}),
            contract_ids=sorted({name_to_id[n] for n in component.contract_names if n in name_to_id}),
        ).model_dump()
        await gold_service.persist_entity_version(
            session,
            entity_type="component",
            entity_id=name_to_id[component.name],
            canonical_name=component.name,
            operation=component.status,
            narrative=component.narrative,
            payload=payload,
            source_component=doc.source_component,
            source_adr_version=doc.version,
            ingestion_date=doc.ingestion_date,
        )
    for contract in result.contracts:
        if contract.action == "unknown":
            continue
        payload = DataContractPayload(
            producer=contract.producer, consumer=contract.consumer, odcs_spec=contract.odcs_spec
        ).model_dump()
        await gold_service.persist_entity_version(
            session,
            entity_type="data_contract",
            entity_id=name_to_id[contract.name],
            canonical_name=contract.name,
            operation=contract.action,
            narrative=contract.narrative,
            payload=payload,
            source_component=doc.source_component,
            source_adr_version=doc.version,
            ingestion_date=doc.ingestion_date,
        )

    architecture_payload = ArchitecturePayload(
        mermaid_diagram=result.mermaid_diagram,
        components=sorted({c.name for c in result.components}),
        dependencies=sorted({dep for c in result.components for dep in c.dependency_names}),
    ).model_dump()
    await gold_service.persist_entity_version(
        session,
        entity_type="architecture",
        entity_id=f"architecture:{doc.source_component}",
        canonical_name=doc.source_component,
        operation=result.architecture_change,
        narrative=result.architecture_narrative,
        payload=architecture_payload,
        source_component=doc.source_component,
        source_adr_version=doc.version,
        ingestion_date=doc.ingestion_date,
    )
    return True


async def main(source_component: str | None, force: bool) -> None:
    async with async_session_factory() as session:
        query = select(SilverDocument)
        if source_component:
            query = query.where(SilverDocument.source_component == source_component)
        docs = (await session.execute(query)).scalars().all()

        processed = 0
        for doc in docs:
            if await _backfill_one(session, doc, force):
                processed += 1
                await session.commit()
        print(f"Backfilled Gold for {processed}/{len(docs)} silver_documents row(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-component", default=None, help="Backfill only this source_component.")
    parser.add_argument(
        "--force", action="store_true", help="Re-extract even if this (source, version) was already processed."
    )
    args = parser.parse_args()
    asyncio.run(main(args.source_component, args.force))
