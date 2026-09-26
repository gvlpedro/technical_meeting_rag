"""Seeds a small, known Gold corpus for `golden_set/questions.json` to ask questions against.

This eval is not about testing extraction (that is `../test_golden_set.py`'s job, which sends
real transcripts through the real graph). It is about testing RETRIEVAL + GENERATION quality
in isolation, against a corpus whose exact contents are known ahead of time — so "did the
right entity come back" and "is the answer faithful to what was retrieved" both have a single,
unambiguous right answer to check against. Depending on whatever happens to already be in a
dev database would make this suite non-reproducible: the golden set's expected answers would
silently drift out of sync with real data changing underneath it.

Four entities, deliberately cross-referencing each other so retrieval has to actually
discriminate between similar, related components — not just find the one thing in an
otherwise-empty tenant:

  - **Payments Gateway** (component, new) — charges the customer.
  - **Checkout Service** (component, modified) — depends on Payments Gateway.
  - **PaymentCharged** (data contract, new) — produced by Payments Gateway, consumed by
    Checkout Service.
  - **Fraud Check Service** (component, new) — screens a transaction before Payments Gateway
    is allowed to charge it; mentions Payments Gateway in its own narrative without being a
    formal dependency, the plausible-distractor case a weaker retrieval strategy could confuse
    with an actual question about Payments Gateway itself.
"""

from datetime import date
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from agents.stages.gold.schemas import ComponentPayload, DataContractPayload
from agents.stages.gold.service import persist_entity_version

_SOURCE_COMPONENT = "chat_eval_seed.en.vtt"
_INGESTION_DATE = date(2026, 1, 15)


async def seed_corpus(session: AsyncSession, tenant: str) -> dict[str, str]:
    """Persists the four entities described in this module's own docstring under `tenant`,
    and returns `{canonical_name: entity_id}` so the test can resolve each golden-set case's
    `expected_canonical_name` to the id retrieval actually has to find."""
    payments_gateway_id = str(uuid4())
    checkout_service_id = str(uuid4())
    payment_charged_id = str(uuid4())
    fraud_check_id = str(uuid4())

    await persist_entity_version(
        session,
        entity_type="component",
        entity_id=payments_gateway_id,
        canonical_name="Payments Gateway",
        operation="new",
        narrative="Payments Gateway is a new component that handles outbound payment charges to external payment providers.",
        payload=ComponentPayload(dependency_ids=[], contract_ids=[payment_charged_id]).model_dump(),
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
        payload=ComponentPayload(dependency_ids=[payments_gateway_id], contract_ids=[payment_charged_id]).model_dump(),
        source_component=_SOURCE_COMPONENT,
        source_adr_version=1,
        ingestion_date=_INGESTION_DATE,
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
    await session.commit()

    return {
        "Payments Gateway": payments_gateway_id,
        "Checkout Service": checkout_service_id,
        "PaymentCharged": payment_charged_id,
        "Fraud Check Service": fraud_check_id,
    }
