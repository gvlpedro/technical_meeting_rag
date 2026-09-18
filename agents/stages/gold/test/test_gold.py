"""Gold layer storage tests, for `agents/stages/gold/service.py`, run against real Postgres.
This file mirrors the style of `tests/test_silver_storage.py`. No LLM is involved anywhere in
this file. Identity resolution, hashing, and versioning are all deterministic, DB-only logic.
See `gold_process.md` §3 and §5. `tests/test_clarification_loop.py`'s Gold-extension test
covers the graph-level, faked-LLM path end to end.
"""

from datetime import date
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from agents.stages.gold.schemas import (
    ArchitecturePayload,
    ComponentPayload,
    DataContractPayload,
    ExtractedDataContract,
)
from agents.stages.gold.service import (
    _reciprocal_rank_fusion,
    already_extracted,
    current_architecture_diagram,
    current_gold_state,
    embed_question,
    ensure_alias,
    entity_history,
    find_entity_by_name_in_text,
    is_evolution_question,
    parse_odcs_spec,
    persist_entity_version,
    resolve_entity_id,
    resolve_entity_id_for_lookup,
    top_k_gold_evolution,
)
from db.models import GoldAlias, GoldEvolution
from db.session import async_session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _unique_component_name() -> str:
    # We use real names, not uuids, so word-boundary-shaped inputs stay realistic. We just
    # make each name collision-free per test run. gold_aliases has no per-test isolation
    # column, so we cannot scope a WHERE clause the way bronze/silver tests scope by
    # ingestion_date.
    return f"Checkout Service {uuid4().hex[:8]}"


async def _cleanup_entity(entity_type: str, entity_id: str) -> None:
    async with async_session_factory() as session:
        await session.execute(
            delete(GoldEvolution).where(GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id)
        )
        await session.execute(
            delete(GoldAlias).where(GoldAlias.entity_type == entity_type, GoldAlias.entity_id == entity_id)
        )
        await session.commit()


def test_component_payload_round_trips_dependency_and_contract_ids():
    payload = ComponentPayload(dependency_ids=["e1", "e2"], contract_ids=["c1"])
    assert payload.model_dump() == {"dependency_ids": ["e1", "e2"], "contract_ids": ["c1"]}


def test_component_payload_defaults_to_empty_lists():
    assert ComponentPayload().model_dump() == {"dependency_ids": [], "contract_ids": []}


def test_data_contract_payload_keeps_odcs_spec_opaque():
    spec = {"apiVersion": "v3.1.0", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
    payload = DataContractPayload(producer="checkout", consumer="billing", odcs_spec=spec)
    assert payload.model_dump()["odcs_spec"] == spec  # It is stored as given, not destructured or validated.


def test_data_contract_payload_defaults_producer_and_consumer_ids_to_empty_string():
    """A contract whose producer/consumer were never resolved — for example, a payload
    persisted before `producer_id`/`consumer_id` existed — must still validate, with an
    honest "not resolved" empty string, not a crash or a made-up id."""
    payload = DataContractPayload(producer="checkout", consumer="billing")
    assert payload.producer_id == ""
    assert payload.consumer_id == ""


def test_data_contract_payload_round_trips_producer_and_consumer_ids():
    producer_id, consumer_id = str(uuid4()), str(uuid4())
    payload = DataContractPayload(
        producer="checkout", consumer="billing", producer_id=producer_id, consumer_id=consumer_id
    )
    assert payload.model_dump()["producer_id"] == producer_id
    assert payload.model_dump()["consumer_id"] == consumer_id


def test_extracted_data_contract_takes_odcs_spec_as_a_string():
    """`ExtractedDataContract` is the `response_format` for the Gold-extraction LLM call.
    It must keep `odcs_spec` as `str`, never `dict`. OpenAI's strict structured-output mode
    rejects any object-typed field that has no `additionalProperties: false`. A deliberately
    open ODCS blob can never declare that. If `odcs_spec` regresses to `dict`, every real
    Gold extraction call that reaches OpenAI breaks. No error appears until the Anthropic
    fallback also fails."""
    contract = ExtractedDataContract(
        name="checkout-events",
        action="new",
        narrative="...",
        producer="checkout",
        consumer="billing",
        odcs_spec='{"apiVersion": "v3.1.0"}',
    )
    assert contract.odcs_spec == '{"apiVersion": "v3.1.0"}'
    with pytest.raises(Exception):
        ExtractedDataContract(
            name="x", action="new", narrative="...", producer="a", consumer="b",
            odcs_spec={"apiVersion": "v3.1.0"},  # A real dict must be rejected here, not silently accepted.
        )


def test_parse_odcs_spec_round_trips_valid_json():
    assert parse_odcs_spec('{"apiVersion": "v3.1.0"}') == {"apiVersion": "v3.1.0"}


def test_parse_odcs_spec_falls_back_to_empty_dict_on_garbage():
    assert parse_odcs_spec("") == {}
    assert parse_odcs_spec("not json") == {}
    assert parse_odcs_spec("[1, 2, 3]") == {}  # This is valid JSON, but not an object. So it still returns {}.


def test_architecture_payload_defaults():
    assert ArchitecturePayload().model_dump() == {
        "mermaid_diagram": "",
        "components": [],
        "dependencies": [],
    }


async def test_resolve_entity_id_exact_match_returns_existing_entity_id():
    entity_type = "component"
    entity_id = str(uuid4())
    name = _unique_component_name()
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            resolved = await resolve_entity_id(session, entity_type, name)
        assert resolved == entity_id
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_resolve_entity_id_fuzzy_match_above_threshold():
    entity_type = "component"
    entity_id = str(uuid4())
    canonical = f"Checkout Service {uuid4().hex[:8]}"
    misspelled = canonical.replace("Checkout", "Checkot")  # This is one typo. It still has more than 0.6 trigram similarity.
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, canonical, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            resolved = await resolve_entity_id(session, entity_type, misspelled)
        assert resolved == entity_id
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_resolve_entity_id_mints_new_id_when_nothing_matches():
    entity_type = "component"
    name = _unique_component_name()
    async with async_session_factory() as session:
        resolved = await resolve_entity_id(session, entity_type, name)
    assert resolved  # This is a fresh ULID string, not empty.
    # This is never inserted anywhere. resolve_entity_id only resolves. ensure_alias inserts.
    async with async_session_factory() as session:
        rows = await session.execute(
            select(GoldAlias).where(GoldAlias.entity_id == resolved)
        )
        assert rows.first() is None


async def test_resolve_entity_id_mints_a_ulid_not_a_uuid4():
    """A freshly minted `entity_id` must be a ULID (`python-ulid`, pinned to 4.0.1): 26
    characters, Crockford-base32 (digits and uppercase letters only, no `I`/`L`/`O`/`U`, and
    no hyphens) — never a `uuid4`'s 36-character hyphenated hex form. This is what lets the
    id double as a short, readable token in a component's own metadata and in a data
    contract's `producer_id`/`consumer_id`, per `resolve_entity_id`'s own docstring."""
    name = _unique_component_name()
    async with async_session_factory() as session:
        resolved = await resolve_entity_id(session, "component", name)
    assert len(resolved) == 26
    assert "-" not in resolved
    assert set(resolved) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


async def test_ensure_alias_is_idempotent():
    entity_type = "component"
    entity_id = str(uuid4())
    name = _unique_component_name()
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)  # This is the same triple, called twice.
            await session.commit()

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldAlias).where(
                        GoldAlias.entity_type == entity_type, GoldAlias.entity_id == entity_id
                    )
                )
            ).all()
        assert len(rows) == 1
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_first_write_is_version_one():
    entity_type = "component"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            version = await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is a new component that handles checkout flows.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()
        assert version == 1
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_same_hash_is_a_noop():
    entity_type = "component"
    entity_id = str(uuid4())
    kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        operation="unchanged",
        narrative="Checkout Service still handles checkout flows the same way.",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
        source_adr_version=1,
        ingestion_date=date(2026, 6, 1),
    )
    try:
        async with async_session_factory() as session:
            first = await persist_entity_version(session, **kwargs)
            await session.commit()
        assert first == 1

        async with async_session_factory() as session:
            second = await persist_entity_version(session, **{**kwargs, "source_adr_version": 2})
            await session.commit()
        assert second is None  # operation, narrative, and payload are identical. So no new row is created.

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).all()
        assert len(rows) == 1
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_different_hash_bumps_version():
    entity_type = "component"
    entity_id = str(uuid4())
    base_kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
        source_adr_version=1,
        ingestion_date=date(2026, 6, 1),
    )
    try:
        async with async_session_factory() as session:
            first = await persist_entity_version(
                session, operation="new", narrative="Checkout Service is new.", **base_kwargs
            )
            await session.commit()
        assert first == 1

        async with async_session_factory() as session:
            second = await persist_entity_version(
                session,
                operation="modified",
                narrative="Checkout Service now also validates payment methods.",
                **{**base_kwargs, "source_adr_version": 2},
            )
            await session.commit()
        assert second == 2

        async with async_session_factory() as session:
            rows = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).all()
        assert len(rows) == 2  # Both versions are kept. The older row stays untouched.
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_already_extracted_true_after_a_write_false_before():
    entity_type = "component"
    entity_id = str(uuid4())
    source = f"meeting-{uuid4().hex[:8]}.en.vtt"
    try:
        async with async_session_factory() as session:
            before = await already_extracted(session, source, 1)
        assert before is False

        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is new.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component=source,
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            after = await already_extracted(session, source, 1)
        assert after is True
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_current_gold_state_returns_only_the_latest_version_per_entity():
    entity_type = "component"
    entity_id = str(uuid4())
    base_kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
        ingestion_date=date(2026, 6, 1),
    )
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session, operation="new", narrative="v1 narrative", source_adr_version=1, **base_kwargs
            )
            await session.commit()
        async with async_session_factory() as session:
            await persist_entity_version(
                session, operation="modified", narrative="v2 narrative", source_adr_version=2, **base_kwargs
            )
            await session.commit()

        async with async_session_factory() as session:
            rows = await current_gold_state(session, entity_type=entity_type)
        matching = [r for r in rows if r.entity_id == entity_id]
        assert len(matching) == 1
        assert matching[0].version == 2
        assert matching[0].narrative == "v2 narrative"
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_current_architecture_diagram_draws_a_node_per_live_component():
    tenant = f"test-arch-{uuid4().hex[:8]}"
    checkout_id, payment_id = str(uuid4()), str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=payment_id,
                canonical_name="Payment Gateway",
                operation="new",
                narrative="Processes card payments.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                tenant=tenant,
            )
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=checkout_id,
                canonical_name="Checkout Service",
                operation="modified",
                narrative="Calls the payment gateway.",
                payload={"dependency_ids": [payment_id], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=2,
                ingestion_date=date(2026, 6, 2),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            diagram = await current_architecture_diagram(session, tenant=tenant)

        assert diagram.startswith("flowchart LR")
        assert "classDef goldNode" in diagram
        # We use backtick markdown-string labels, never raw <br/>/<sub> HTML. See the
        # function's own docstring for why. Those HTML tags broke node visibility under
        # Streamlit's strict security mode.
        assert "<br/>" not in diagram and "<sub>" not in diagram
        assert '["`Payment Gateway\n*meeting v1*`"]' in diagram
        assert '["`Checkout Service\n*meeting v2*`"]' in diagram
        assert "-->" in diagram  # Checkout Service depends on Payment Gateway
        # There is no `click ... href` directive here. We confirmed, with a real headless-Chrome
        # run against `st.mermaid_chart` and not just mermaid-cli, that this directive makes
        # Streamlit render an empty node group while still drawing edges. That is exactly the
        # "only edges" bug. ADR navigation happens through the plain link table that
        # `_architecture_history_tab` renders instead.
        assert "click" not in diagram
        assert "class n0,n1 goldNode" in diagram
    finally:
        await _cleanup_entity("component", checkout_id)
        await _cleanup_entity("component", payment_id)


async def test_current_architecture_diagram_is_empty_when_gold_has_no_live_component():
    tenant = f"test-arch-empty-{uuid4().hex[:8]}"
    async with async_session_factory() as session:
        diagram = await current_architecture_diagram(session, tenant=tenant)
    assert diagram == ""


async def test_current_architecture_diagram_drops_a_removed_component():
    tenant = f"test-arch-{uuid4().hex[:8]}"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=entity_id,
                canonical_name="Legacy Notifier",
                operation="removed",
                narrative="Decommissioned.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                tenant=tenant,
            )
            await session.commit()

        async with async_session_factory() as session:
            diagram = await current_architecture_diagram(session, tenant=tenant)
        assert diagram == ""
    finally:
        await _cleanup_entity("component", entity_id)


async def test_persist_entity_version_stores_authored_by():
    """`authored_by` copies forward the same value `silver_documents.authored_by` already
    holds — see `db/models.py`'s `GoldEvolution.authored_by` column comment. This is what lets
    an evolution timeline answer say who introduced or changed an entity."""
    entity_type = "component"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is new.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                authored_by="alex",
            )
            await session.commit()

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).scalar_one()
        assert row.authored_by == "alex"
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_persist_entity_version_defaults_authored_by_to_empty_string():
    """A CLI, script, or test run with no real logged-in user must not crash or store `None` —
    the same `""` default `silver_documents.authored_by` and `bronze_documents.uploaded_by`
    already use."""
    entity_type = "component"
    entity_id = str(uuid4())
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type=entity_type,
                entity_id=entity_id,
                canonical_name="Checkout Service",
                operation="new",
                narrative="Checkout Service is new.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            await session.commit()

        async with async_session_factory() as session:
            row = (
                await session.execute(
                    select(GoldEvolution).where(
                        GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id
                    )
                )
            ).scalar_one()
        assert row.authored_by == ""
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_resolve_entity_id_for_lookup_returns_none_when_nothing_matches():
    """Unlike `resolve_entity_id`, the read-only lookup must never mint a fresh id for "no
    match" — a caller asking "does this entity exist" needs an honest `None`, not an empty,
    alias-less identity that would silently read as "found it, but no history"."""
    name = _unique_component_name()
    async with async_session_factory() as session:
        resolved = await resolve_entity_id_for_lookup(session, "component", name)
    assert resolved is None


async def test_resolve_entity_id_for_lookup_returns_existing_id_on_exact_match():
    entity_type = "component"
    entity_id = str(uuid4())
    name = _unique_component_name()
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            resolved = await resolve_entity_id_for_lookup(session, entity_type, name)
        assert resolved == entity_id
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_entity_history_returns_every_version_oldest_first():
    entity_type = "component"
    entity_id = str(uuid4())
    base_kwargs = dict(
        entity_type=entity_type,
        entity_id=entity_id,
        canonical_name="Checkout Service",
        payload={"dependency_ids": [], "contract_ids": []},
        source_component="meeting.en.vtt",
    )
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                operation="new",
                narrative="Introduced at launch.",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
                authored_by="alex",
                **base_kwargs,
            )
            await session.commit()
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                operation="modified",
                narrative="Added payment validation.",
                source_adr_version=2,
                ingestion_date=date(2026, 7, 1),
                authored_by="sam",
                **base_kwargs,
            )
            await session.commit()

        async with async_session_factory() as session:
            history = await entity_history(session, entity_type, entity_id)
        assert [row.version for row in history] == [1, 2]  # oldest first, not newest first
        assert history[0].authored_by == "alex"
        assert history[1].authored_by == "sam"
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_entity_history_is_empty_for_an_entity_never_seen():
    async with async_session_factory() as session:
        history = await entity_history(session, "component", str(uuid4()))
    assert history == []


async def test_find_entity_by_name_in_text_matches_a_known_alias():
    entity_type = "component"
    entity_id = str(uuid4())
    name = f"Risk Notification Service {uuid4().hex[:8]}"
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, entity_type, entity_id, name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            match = await find_entity_by_name_in_text(
                session, f"What evolution has {name} had over time?"
            )
        assert match == (entity_type, entity_id, name)
    finally:
        await _cleanup_entity(entity_type, entity_id)


async def test_find_entity_by_name_in_text_returns_none_when_no_alias_appears():
    async with async_session_factory() as session:
        match = await find_entity_by_name_in_text(session, "What is the weather like today?")
    assert match is None


async def test_find_entity_by_name_in_text_prefers_the_longest_match():
    """"Order Service" and "Order" both being aliases (for two different entities) is a real
    scenario, not a contrived one: a short, generic fragment can end up registered as its own
    alias variant. The longer, more specific match wins — see the function's own docstring for
    why a short alias is too likely to be a coincidence."""
    short_id, long_id = str(uuid4()), str(uuid4())
    suffix = uuid4().hex[:8]
    short_name = f"Order{suffix}"
    long_name = f"Order{suffix} Fulfillment Service"
    try:
        async with async_session_factory() as session:
            await ensure_alias(session, "component", short_id, short_name, "meeting.en.vtt", 1)
            await ensure_alias(session, "component", long_id, long_name, "meeting.en.vtt", 1)
            await session.commit()

        async with async_session_factory() as session:
            match = await find_entity_by_name_in_text(
                session, f"How has {long_name} evolved over time?"
            )
        assert match == ("component", long_id, long_name)
    finally:
        await _cleanup_entity("component", short_id)
        await _cleanup_entity("component", long_id)


def test_is_evolution_question_recognizes_english_and_spanish_phrasing():
    assert is_evolution_question("How has the Order Service evolved over time?")
    assert is_evolution_question("What is the history of the checkout component?")
    assert is_evolution_question("¿Qué evolución ha tenido el componente X a lo largo del tiempo?")
    assert is_evolution_question("Dame la historia del componente de pagos.")


def test_is_evolution_question_is_false_for_an_ordinary_factual_question():
    assert not is_evolution_question("Who is the producer of the OrderCreated contract?")
    assert not is_evolution_question("What is the current status of the Checkout Service?")


# =====================================================================================
# Hybrid search (vector + lexical, fused with RRF) — see `.tmp/advanced_techniques.md` §1
# =====================================================================================


def test_reciprocal_rank_fusion_lets_a_strong_lexical_match_outrank_a_lone_vector_top_hit():
    """id 7 ranks only 7th in the vector ranking, but 1st (and only) in the lexical one.
    `1/(60+7) + 1/(60+1) ≈ 0.0313` beats id 1's `1/(60+1) ≈ 0.0164` — a lone rank-1 hit in one
    signal, with zero support from the other. This is the concrete case hybrid search exists
    for: a moderate embedding match that is ALSO the best lexical match should outrank a
    perfect embedding match that no lexical signal agrees with."""
    vector_ranking = list(range(1, 11))  # id 7 sits at position 7
    lexical_ranking = [7]
    fused = _reciprocal_rank_fusion([vector_ranking, lexical_ranking])
    assert fused[0] == 7


def test_reciprocal_rank_fusion_id_absent_from_a_ranking_gets_no_penalty_from_it():
    """An id missing from one input ranking contributes nothing from that ranking — never a
    negative score. Two ids that each appear in exactly one ranking, both at rank 1, must tie
    (both score `1/(60+1)`), not have the "missing" one penalized below a real rank."""
    fused = _reciprocal_rank_fusion([[1], [2]])
    assert set(fused) == {1, 2}


def test_reciprocal_rank_fusion_returns_empty_for_no_candidates():
    assert _reciprocal_rank_fusion([]) == []
    assert _reciprocal_rank_fusion([[], []]) == []


async def test_top_k_gold_evolution_hybrid_mode_finds_a_lexical_match_a_bad_vector_search_misses():
    """This proves the lexical branch actually rescues a row the vector branch alone would
    never surface — not just that the function runs. The query vector is deliberately set to
    the EXACT embedding of an unrelated distractor row (cosine distance 0), so `mode="vector"`
    is guaranteed to return only that distractor at `k=1`. `mode="hybrid"`, given the SAME
    adversarial vector but the real question text, must still surface the target row through
    `ts_rank` + RRF."""
    target_id, distractor_id = str(uuid4()), str(uuid4())
    unique_token = f"xzqv{uuid4().hex[:10]}"
    adversarial_vector = await embed_question("Completely unrelated text about weather forecasting.")
    try:
        async with async_session_factory() as session:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=target_id,
                canonical_name=f"{unique_token} Adapter",
                operation="new",
                narrative=f"{unique_token} is a proprietary internal protocol adapter.",
                payload={"dependency_ids": [], "contract_ids": []},
                source_component="meeting.en.vtt",
                source_adr_version=1,
                ingestion_date=date(2026, 6, 1),
            )
            # A hand-built row whose embedding is IDENTICAL to the query vector — guaranteed
            # cosine distance 0, so it always wins rank 1 under mode="vector", regardless of
            # whatever else exists in this test database.
            session.add(
                GoldEvolution(
                    tenant="default",
                    entity_type="component",
                    entity_id=distractor_id,
                    canonical_name="Weather Distractor",
                    version=1,
                    operation="new",
                    narrative="Completely unrelated text about weather forecasting.",
                    payload={},
                    embedding=adversarial_vector,
                    entity_hash="distractor",
                    source_component="meeting.en.vtt",
                    source_adr_version=1,
                    ingestion_date=date(2026, 6, 1),
                )
            )
            await session.commit()

        async with async_session_factory() as session:
            vector_only = await top_k_gold_evolution(session, adversarial_vector, k=1, mode="vector")
            hybrid = await top_k_gold_evolution(
                session, adversarial_vector, k=5, mode="hybrid", question_text=f"What is {unique_token}?"
            )
        assert {r.entity_id for r in vector_only} == {distractor_id}
        assert target_id in {r.entity_id for r in hybrid}
    finally:
        await _cleanup_entity("component", target_id)
        await _cleanup_entity("component", distractor_id)


async def test_top_k_gold_evolution_hybrid_mode_requires_question_text():
    # `session=None`: both validation errors below raise before the function ever touches
    # `session`, so no real database session is needed to prove them.
    with pytest.raises(ValueError, match="question_text"):
        await top_k_gold_evolution(None, [0.0], mode="hybrid")


async def test_top_k_gold_evolution_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown"):
        await top_k_gold_evolution(None, [0.0], mode="bm25")
