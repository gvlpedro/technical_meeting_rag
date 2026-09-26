"""Seeds a small, known Gold corpus for `golden_set/questions.json` to ask questions against.

This eval is not about testing extraction (that is `../test_golden_set.py`'s job, which sends
real transcripts through the real graph). It is about testing RETRIEVAL + GENERATION quality
in isolation, against a corpus whose exact contents are known ahead of time — so "did the
right entity come back" and "is the answer faithful to what was retrieved" both have a single,
unambiguous right answer to check against. Depending on whatever happens to already be in a
dev database would make this suite non-reproducible: the golden set's expected answers would
silently drift out of sync with real data changing underneath it.

Six entities across two ingestion dates, deliberately cross-referencing each other so retrieval
has to actually discriminate between similar, related components — not just find the one thing
in an otherwise-empty tenant. The two dates also give the lineage/timeline questions in
`golden_set/questions.json` (`.tmp/improve_timeline_questions_and_linage.md`) something real to
find: a `valid_to` window that actually closed, and a successor introduced later than the rest:

  - **Payments Gateway** (component, new, 2026-01-15) — charges the customer. Produces
    (`output_contract_ids`) PaymentCharged.
  - **Checkout Service** (component, 2026-01-15 then modified again 2026-03-01) — depends on
    Payments Gateway; consumes (`input_contract_ids`) PaymentCharged. Two versions on purpose:
    v1's `valid_to` closes to v2's `ingestion_date`, so an evolution question about it has a
    real "in effect until..." to report, not just a single always-current step.
  - **PaymentCharged** (data contract, new, 2026-01-15) — produced by Payments Gateway, consumed
    by Checkout Service.
  - **Fraud Check Service** (component, new, 2026-01-15) — screens a transaction before Payments
    Gateway is allowed to charge it; mentions Payments Gateway in its own narrative without being
    a formal dependency, the plausible-distractor case a weaker retrieval strategy could confuse
    with an actual question about Payments Gateway itself.
  - **Refund Service** (component, new, 2026-03-01) — a SECOND, later-introduced dependent of
    Payments Gateway. Together with Checkout Service, this is what makes "which components
    depend on Payments Gateway" a real two-answer successors question, not a trivially single
    one `dependency_ids` alone could already answer without `get_successors`.
"""

from datetime import date
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from agents.stages.gold.schemas import ComponentPayload, DataContractPayload
from agents.stages.gold.service import ensure_alias, persist_entity_version

_SOURCE_COMPONENT = "chat_eval_seed.en.vtt"
_INGESTION_DATE = date(2026, 1, 15)
_LATER_INGESTION_DATE = date(2026, 3, 1)


async def seed_corpus(session: AsyncSession, tenant: str) -> dict[str, str]:
    """Persists the four entities described in this module's own docstring under `tenant`,
    and returns `{canonical_name: entity_id}` so the test can resolve each golden-set case's
    `expected_canonical_name` to the id retrieval actually has to find."""
    payments_gateway_id = str(uuid4())
    checkout_service_id = str(uuid4())
    payment_charged_id = str(uuid4())
    fraud_check_id = str(uuid4())
    refund_service_id = str(uuid4())

    await persist_entity_version(
        session,
        entity_type="component",
        entity_id=payments_gateway_id,
        canonical_name="Payments Gateway",
        operation="new",
        narrative="Payments Gateway is a new component that handles outbound payment charges to external payment providers.",
        payload=ComponentPayload(
            dependency_ids=[], contract_ids=[payment_charged_id], output_contract_ids=[payment_charged_id]
        ).model_dump(),
        source_component=_SOURCE_COMPONENT,
        source_adr_version=1,
        ingestion_date=_INGESTION_DATE,
        tenant=tenant,
    )
    await persist_entity_version(
        session,
        entity_type="component",
        entity_id=checkout_service_id,
        canonical_name="Checkout Service",
        operation="modified",
        narrative="Checkout Service was modified to call Payments Gateway to charge the customer once an order is placed.",
        payload=ComponentPayload(
            dependency_ids=[payments_gateway_id], contract_ids=[payment_charged_id],
            input_contract_ids=[payment_charged_id],
        ).model_dump(),
        source_component=_SOURCE_COMPONENT,
        source_adr_version=1,
        ingestion_date=_INGESTION_DATE,
        tenant=tenant,
    )
    # A second version of the same entity — this is what gives `valid_to` a real window to
    # close (v1's closes to this row's `ingestion_date`) and the evolution question in
    # `golden_set/questions.json` a genuine two-step history to narrate.
    await persist_entity_version(
        session,
        entity_type="component",
        entity_id=checkout_service_id,
        canonical_name="Checkout Service",
        operation="modified",
        narrative="Checkout Service was modified again to also emit a refund event when a customer requests a refund.",
        payload=ComponentPayload(
            dependency_ids=[payments_gateway_id], contract_ids=[payment_charged_id],
            input_contract_ids=[payment_charged_id],
        ).model_dump(),
        source_component=_SOURCE_COMPONENT,
        source_adr_version=2,
        ingestion_date=_LATER_INGESTION_DATE,
        tenant=tenant,
    )
    await persist_entity_version(
        session,
        entity_type="data_contract",
        entity_id=payment_charged_id,
        canonical_name="PaymentCharged",
        operation="new",
        narrative="PaymentCharged is a new data contract produced by Payments Gateway and consumed by Checkout Service, emitted once a charge succeeds.",
        payload=DataContractPayload(
            producer="Payments Gateway",
            consumer="Checkout Service",
            producer_id=payments_gateway_id,
            consumer_id=checkout_service_id,
        ).model_dump(),
        source_component=_SOURCE_COMPONENT,
        source_adr_version=1,
        ingestion_date=_INGESTION_DATE,
        tenant=tenant,
    )
    await persist_entity_version(
        session,
        entity_type="component",
        entity_id=fraud_check_id,
        canonical_name="Fraud Check Service",
        operation="new",
        narrative="Fraud Check Service is a new component that screens a transaction for fraud before Payments Gateway is allowed to charge it.",
        payload=ComponentPayload(dependency_ids=[], contract_ids=[]).model_dump(),
        source_component=_SOURCE_COMPONENT,
        source_adr_version=1,
        ingestion_date=_INGESTION_DATE,
        tenant=tenant,
    )
    await persist_entity_version(
        session,
        entity_type="component",
        entity_id=refund_service_id,
        canonical_name="Refund Service",
        operation="new",
        narrative="Refund Service is a new component, introduced later, that calls Payments Gateway to issue refunds back to the customer.",
        payload=ComponentPayload(dependency_ids=[payments_gateway_id], contract_ids=[]).model_dump(),
        source_component=_SOURCE_COMPONENT,
        source_adr_version=2,
        ingestion_date=_LATER_INGESTION_DATE,
        tenant=tenant,
    )
    # `persist_entity_version` alone never writes `gold_aliases` — that only happens through
    # `resolve_and_alias`, one step earlier in the real pipeline (`resolve_gold_identity`),
    # which this synthetic seed skips entirely. `find_entity_by_name_in_text` (the entity-name
    # lookup the evolution-question path uses) reads `gold_aliases`, not `GoldEvolution`
    # directly — without these, "checkout-service-evolution" in the golden set could never
    # resolve, no matter how it is phrased.
    for canonical_name, entity_id in (
        ("Payments Gateway", payments_gateway_id),
        ("Checkout Service", checkout_service_id),
        ("Fraud Check Service", fraud_check_id),
        ("Refund Service", refund_service_id),
    ):
        await ensure_alias(
            session, "component", entity_id, canonical_name, _SOURCE_COMPONENT, 1, tenant=tenant
        )
    await session.commit()

    return {
        "Payments Gateway": payments_gateway_id,
        "Checkout Service": checkout_service_id,
        "PaymentCharged": payment_charged_id,
        "Fraud Check Service": fraud_check_id,
        "Refund Service": refund_service_id,
    }
