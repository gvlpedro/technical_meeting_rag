"""Reusable Gold-stage logic. Callable directly, for example by `scripts/backfill_gold.py` or
tests, without the full LangGraph state. The three Gold nodes in `agents/graph.py`
(`extract_gold_facts`, `resolve_gold_identity`, `persist_gold_evolution`) are thin wrappers
around the functions here.

Gold's per-ADR reconciliation runs inside Silver's own graph run, not as a separate pipeline
watching `silver_documents` for changes.

Every function below takes explicit arguments, never a graph `state`. This lets
`scripts/backfill_gold.py` call the same functions the graph does, replaying Gold over
existing `silver_documents` rows without re-running Silver's own loop."""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal

from sqlalchemy import Numeric, and_, func, or_, select, text, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from agents.shared import extract_authors_line
from agents.stages.gold.prompts import build_gold_extraction_prompt
from agents.stages.gold.retrieval.citation_verification import _verify_citations
from agents.stages.gold.retrieval.conversational_memory import MAX_HISTORY_MESSAGES, _format_history_block
from agents.stages.gold.retrieval.corrective_rag import retrieve_with_correction
from agents.stages.gold.retrieval.deduplication import _dedupe_ids_by_entity
from agents.stages.gold.retrieval.hybrid_search import (
    RRF_K_CONSTANT,
    _reciprocal_rank_fusion,
    _top_k_ids_by_lexical_rank,
    _top_k_ids_by_vector,
)
from agents.stages.gold.retrieval.query_expansion import expand_question
from agents.stages.gold.retrieval.reranking import _rerank_ids
from agents.stages.gold.schemas import (
    ArchitecturePayload,
    ComponentPayload,
    DataContractPayload,
    GoldEntityType,
    GoldExtractionResult,
    GoldOperation,
    GroundedAnswer,
)
from agents.template import load_json_response
from db.models import GoldAlias, GoldEvolution
from ingestion.embedder import embed
from llm import router

# `resolve_entity_id` uses this threshold for a fuzzy match. Lower risks merging genuinely
# different names ("Order Service" and "Orders API"). Higher risks minting a duplicate
# entity_id for a trivial spelling drift. Not yet tuned against a real corpus.
FUZZY_MATCH_THRESHOLD = 0.6


def content_hash(content: str) -> str:
    """Turns a string into a version hash. Shared by `_entity_hash` and
    `agents.graph._persist_document_version`."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_odcs_spec(raw: str) -> dict:
    """Parses `ExtractedDataContract.odcs_spec`'s JSON string back into the dict
    `DataContractPayload.odcs_spec` stores. Shared by `extract_and_persist_gold_facts` and
    `agents.graph._persist_contracts`.

    Falls back to `{}` for anything that is not a JSON object, so one malformed spec never
    crashes the whole ADR's Gold persistence."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def latest_odcs_spec(session: AsyncSession, entity_id: str, tenant: str = "default") -> dict:
    """The most recently persisted `odcs_spec` for this data contract, or `{}` if it has never
    been persisted before. A later ADR that marks a contract `unchanged` naturally does not
    restate its full schema, so `parse_odcs_spec` on that ADR's own extraction comes back `{}` —
    both real callers (`extract_and_persist_gold_facts` below, `agents.graph._persist_contracts`)
    call this to carry the previous spec forward instead of persisting an empty one over a real
    one. See `.tmp/improve_timeline_questions_and_linage.md` §5 — this is the blocking
    prerequisite for the input/output contract split to show anything meaningful."""
    latest = (
        await session.execute(
            select(GoldEvolution)
            .where(
                GoldEvolution.tenant == tenant,
                GoldEvolution.entity_type == "data_contract",
                GoldEvolution.entity_id == entity_id,
            )
            .order_by(GoldEvolution.version.desc())
            .limit(1)
        )
    ).scalars().first()
    if latest is None:
        return {}
    return DataContractPayload.model_validate(latest.payload).odcs_spec


def classify_contract_directions(
    contracts: list, name_to_id: dict[str, str]
) -> dict[str, dict[str, set[str]]]:
    """Maps each component name to the contract ids it consumes ("input") and produces
    ("output"), from this same ADR's own extraction. Shared by `extract_and_persist_gold_facts`
    and `agents.graph._persist_components`. `contracts` must be a list of plain dicts, not
    `ExtractedDataContract` instances, since this indexes with `contract["field"]`."""
    directions: dict[str, dict[str, set[str]]] = {}
    for contract in contracts:
        key = memo_key("data_contract", contract["name"])
        if contract["action"] == "unknown" or key not in name_to_id:
            continue
        contract_id = name_to_id[key]
        consumer, producer = contract.get("consumer"), contract.get("producer")
        if consumer:
            directions.setdefault(consumer, {"input": set(), "output": set()})["input"].add(contract_id)
        if producer:
            directions.setdefault(producer, {"input": set(), "output": set()})["output"].add(contract_id)
    return directions


def contracts_with_real_changes(contracts: list, name_to_id: dict[str, str]) -> set[str]:
    """Entity ids of every contract in this extraction whose `action` is a real change (not
    `"unchanged"`/`"unknown"`). Used to correct a component's own `"unchanged"` to `"modified"`
    when a contract it already depends on changed this round. Same "plain dicts" contract as
    `classify_contract_directions`; both are called together on the same `contracts` list."""
    return {
        name_to_id[memo_key("data_contract", c["name"])]
        for c in contracts
        if c["action"] not in ("unchanged", "unknown") and memo_key("data_contract", c["name"]) in name_to_id
    }


def _entity_hash(operation: GoldOperation, narrative: str, payload: dict) -> str:
    """Same hash-compare-then-bump mechanism as `content_hash`, scoped to one entity instead of
    a whole document.

    `payload` is serialized with `sort_keys=True`, so two identical payloads hash the same even
    with keys in a different order. This does NOT sort list elements. Callers must already sort
    list fields like `dependency_ids` before calling this, or an unordered list can trigger a
    version bump that should not happen."""
    canonical_payload = json.dumps(payload, sort_keys=True)
    return content_hash(f"{operation}\n{narrative}\n{canonical_payload}")


async def extract_gold_facts_for_source(adr_content: str) -> GoldExtractionResult:
    """One structured-extraction LLM call, at `temperature=0`: deterministic extraction, not
    creative writing. `reasoning_effort="none"` turns off a reasoning-locked model's own
    temperature check. Anthropic's reasoning-locked models have no such setting and require
    temperature=1; a fallback to Anthropic here would fail instead. Accepted risk, not fixed.

    Takes only `adr_content`, never a pre-given component/contract list. Gold discovers every
    component and contract from the final, clarified ADR."""
    messages = build_gold_extraction_prompt(adr_content)
    response = await router.complete(
        messages, response_format=GoldExtractionResult, temperature=0, reasoning_effort="none"
    )
    return GoldExtractionResult.model_validate(load_json_response(response.choices[0].message.content))


async def already_extracted(
    session: AsyncSession, source_component: str, source_adr_version: int, *, tenant: str = "default"
) -> bool:
    """Returns `True` if Gold has already written at least one `gold_evolution` row for this
    exact `(tenant, source_component, source_adr_version)` triple. That means this specific
    version of this ADR was already reconciled, so there is nothing to re-extract.

    This approach is simpler than tracking a separate "last processed content_hash" column. That
    was an earlier draft of this idea (`.tmp/gold_process_v5.md` §3). `source_adr_version` is
    already the output of Silver's own hash-based versioning. So an unchanged ADR never gets a
    new version to begin with. Checking whether THIS version was processed does the same job and
    needs no extra column.

    This function filters by `tenant` because two tenants can legitimately upload a file with the
    exact same name and reach the exact same version number, independently of each other.
    Without this filter, tenant B's first-ever extraction could get skipped, just because tenant A
    already did "the same" (source_component, version) pair."""
    result = await session.execute(
        select(GoldEvolution.id)
        .where(
            GoldEvolution.tenant == tenant,
            GoldEvolution.source_component == source_component,
            GoldEvolution.source_adr_version == source_adr_version,
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def gold_entities_for_adr(
    session: AsyncSession, source_component: str, source_adr_version: int, *, tenant: str = "default"
) -> Sequence[GoldEvolution]:
    """Returns every `gold_evolution` row this `(tenant, source_component, source_adr_version)`
    triple actually wrote — only entities this ADR version genuinely added or changed. Backs
    the frontend's "what did this ADR generate in Gold" view."""
    result = await session.execute(
        select(GoldEvolution)
        .where(
            GoldEvolution.tenant == tenant,
            GoldEvolution.source_component == source_component,
            GoldEvolution.source_adr_version == source_adr_version,
        )
        .order_by(GoldEvolution.entity_type, GoldEvolution.canonical_name)
    )
    return result.scalars().all()


async def _lookup_entity_id(
    session: AsyncSession, entity_type: GoldEntityType, name: str, *, tenant: str
) -> str | None:
    """Shared exact-then-fuzzy lookup for `resolve_entity_id` and `resolve_entity_id_for_lookup`.
    Tries an exact match on `gold_aliases.alias` first, then a `pg_trgm` fuzzy match above
    `FUZZY_MATCH_THRESHOLD`. Returns `None` if neither matches.

    `tenant` scopes both lookups, or two different companies' same-named components would
    merge into one entity_id.

    Rounds `similarity()` to 4 decimals before comparing. Comparing the raw `real` to a Python
    `float` widens it first, which can push an exact 0.6 just over the threshold — confirmed:
    this silently merged two unrelated "frontend" components."""
    exact = (
        await session.execute(
            select(GoldAlias.entity_id).where(
                GoldAlias.tenant == tenant, GoldAlias.entity_type == entity_type, GoldAlias.alias == name
            )
        )
    ).scalars().first()
    if exact is not None:
        return exact

    rounded_similarity = func.round(func.similarity(GoldAlias.alias, name).cast(Numeric), 4)
    fuzzy = (
        await session.execute(
            select(GoldAlias.entity_id)
            .where(
                GoldAlias.tenant == tenant,
                GoldAlias.entity_type == entity_type,
                rounded_similarity > FUZZY_MATCH_THRESHOLD,
            )
            .order_by(rounded_similarity.desc())
            .limit(1)
        )
    ).scalars().first()
    return fuzzy


async def resolve_entity_id(
    session: AsyncSession, entity_type: GoldEntityType, name: str, *, tenant: str = "default"
) -> str:
    """Write-path identity resolution for `resolve_and_alias`. Tries `_lookup_entity_id` first.
    Mints a new `entity_id` as a ULID string if nothing matches — opaque, but short and
    readable, unlike a `uuid4`.

    Does not insert the alias row itself; see `ensure_alias`. Never use this for a read-only
    lookup — it mints a fresh id even for "no match." Use `resolve_entity_id_for_lookup`
    there."""
    existing = await _lookup_entity_id(session, entity_type, name, tenant=tenant)
    return existing if existing is not None else str(ULID())


async def resolve_entity_id_for_lookup(
    session: AsyncSession, entity_type: GoldEntityType, name: str, *, tenant: str = "default"
) -> str | None:
    """Read-only counterpart to `resolve_entity_id`: asks "does this entity exist," never mints
    a fresh id. Used by `find_entity_by_name_in_text` to resolve a name before fetching its
    history."""
    return await _lookup_entity_id(session, entity_type, name, tenant=tenant)


async def ensure_alias(
    session: AsyncSession,
    entity_type: GoldEntityType,
    entity_id: str,
    alias: str,
    source_component: str,
    source_adr_version: int,
    *,
    tenant: str = "default",
) -> None:
    """Inserts this `(tenant, entity_type, entity_id, alias)` row if it does not exist, with
    `ON CONFLICT DO NOTHING` — the same idempotent-insert convention used elsewhere in this
    pipeline."""
    stmt = (
        insert(GoldAlias)
        .values(
            tenant=tenant,
            entity_type=entity_type,
            entity_id=entity_id,
            alias=alias,
            source_component=source_component,
            source_adr_version=source_adr_version,
        )
        .on_conflict_do_nothing(index_elements=["tenant", "entity_type", "entity_id", "alias"])
    )
    await session.execute(stmt)


def memo_key(entity_type: GoldEntityType, name: str) -> str:
    """Builds a `name_to_id` memo key. Every caller that reads this dict must use this, never
    a bare `name`.

    Fixes a real bug: a component and a data contract can share the same name ("Catalog
    Listing"). Keying by bare name let the second type reuse the first type's `entity_id`,
    splitting that contract's history across two identities. Keying by `(entity_type, name)`
    stops that."""
    return f"{entity_type}:{name}"


async def resolve_and_alias(
    session: AsyncSession,
    entity_type: GoldEntityType,
    name: str,
    name_to_id: dict[str, str],
    source_component: str,
    source_adr_version: int,
    *,
    tenant: str = "default",
) -> str:
    """Composes `resolve_entity_id` and `ensure_alias`, memoized against `name_to_id`, keyed by
    `memo_key(entity_type, name)`. A name already resolved this batch is just a dict lookup.

    Shared by `agents.graph.resolve_gold_identity` and `scripts.backfill_gold._backfill_one`."""
    key = memo_key(entity_type, name)
    if key in name_to_id:
        return name_to_id[key]
    entity_id = await resolve_entity_id(session, entity_type, name, tenant=tenant)
    name_to_id[key] = entity_id
    await ensure_alias(session, entity_type, entity_id, name, source_component, source_adr_version, tenant=tenant)
    return entity_id


async def persist_entity_version(
    session: AsyncSession,
    *,
    entity_type: GoldEntityType,
    entity_id: str,
    canonical_name: str,
    operation: GoldOperation,
    narrative: str,
    payload: dict,
    source_component: str,
    source_adr_version: int,
    ingestion_date: date,
    tenant: str = "default",
    authored_by: str = "",
    force_new_version: bool = False,
) -> int | None:
    """Hash-compare-then-bump against the latest `gold_evolution` row for this
    `(entity_type, entity_id)`. Same mechanism as `agents.graph._persist_document_version`,
    scoped to one entity. Identical `entity_hash` to the latest row is a no-op (`None`).
    Otherwise inserts `version + 1` (or 1 for the first row), embedding `narrative`.

    `authored_by` is not part of `_entity_hash`: it describes who wrote this version, not
    what changed.

    Caller must never call this with `operation == "unknown"` — see
    `agents.graph.persist_gold_evolution`'s skip step.

    See `doc/cicle_evolution.md` for the full rules. Four live here:

    1. **Payload corrects a false "unchanged" for a component.** If a component's payload
       differs from the previous version even though its narrated status says "unchanged",
       `operation` is corrected to `"modified"` before persisting. Does not apply to
       `data_contract`: `ContractAction` has no generic "modified".
    2. **A carried-forward narrative neutralizes re-wording noise.** When `operation ==
       "unchanged"` and a prior version exists, `narrative` is replaced with the prior
       version's own narrative, both for hashing and for storage — otherwise a fresh re-wording
       of "still unchanged" creates spurious versions. Fixed a real bug: several contracts
       picked up 2-3 spurious versions from wording drift alone.
    3. **The no-op comparison hashes against `latest.operation`, never the literal
       "unchanged".** `latest.entity_hash` was computed with whatever operation actually
       produced it. Comparing against the literal string "unchanged" instead was a real bug:
       the first "unchanged" ADR after a real change always missed the no-op check and
       inserted a bogus version stamped with that ADR's own source — a clicked diagram node
       then linked to the wrong ADR. `operation` itself is still stored as "unchanged"; only
       the hash comparison substitutes `latest.operation`.
    4. **`force_new_version` inserts a new version even on a hash match.** Used by
       `apply_contract_ripple_effect`: a component whose own content never changed still needs
       a version stamped with THIS ADR, because something it depends on changed instead."""
    latest = (
        await session.execute(
            select(GoldEvolution)
            .where(
                GoldEvolution.tenant == tenant,
                GoldEvolution.entity_type == entity_type,
                GoldEvolution.entity_id == entity_id,
            )
            .order_by(GoldEvolution.version.desc())
            .limit(1)
        )
    ).scalars().first()

    if operation == "unchanged" and latest is not None and entity_type == "component":
        payload_changed = json.dumps(payload, sort_keys=True) != json.dumps(latest.payload, sort_keys=True)
        if payload_changed:
            operation = "modified"

    hash_operation = operation
    if operation == "unchanged" and latest is not None:
        # Replace both the hashed AND stored narrative, so the chain stays consistent. See rule 2.
        narrative = latest.narrative
        # Hash against latest.operation, not the literal "unchanged" — see rule 3.
        hash_operation = latest.operation

    entity_hash = _entity_hash(hash_operation, narrative, payload)

    if not force_new_version and latest is not None and latest.entity_hash == entity_hash:
        return None

    if latest is not None:
        # Closes the outgoing version's validity window now that it is superseded.
        latest.valid_to = ingestion_date

    version = 1 if latest is None else latest.version + 1
    [embedding] = await asyncio.to_thread(embed, [narrative])
    session.add(
        GoldEvolution(
            tenant=tenant,
            entity_type=entity_type,
            entity_id=entity_id,
            canonical_name=canonical_name,
            version=version,
            operation=operation,
            narrative=narrative,
            payload=payload,
            embedding=embedding,
            entity_hash=entity_hash,
            source_component=source_component,
            source_adr_version=source_adr_version,
            ingestion_date=ingestion_date,
            authored_by=authored_by,
        )
    )
    return version


async def apply_contract_ripple_effect(
    session: AsyncSession,
    *,
    changed_contract_ids: set[str],
    contract_directions_by_component_id: dict[str, dict[str, set[str]]],
    mentioned_component_ids: set[str],
    source_component: str,
    source_adr_version: int,
    ingestion_date: date,
    tenant: str = "default",
) -> None:
    """Applies `doc/cicle_evolution.md`'s "Regla especial", computed deterministically against
    `gold_evolution` instead of requiring the LLM to re-mention an unchanged component.

    Checks every LIVE component this ADR's extraction did NOT already persist a version for
    (`mentioned_component_ids`), against two conditions:

    1. **Starts producing/consuming a contract it never had before.**
       `contract_directions_by_component_id.get(row.entity_id)` is non-empty. This is a real
       change to the component's own payload, so the new ids are merged in and persisted
       normally — the hash will differ, no `force_new_version` needed.
    2. **A contract it already had is the one that changed.** Its last-persisted
       `input_contract_ids`/`output_contract_ids` intersects `changed_contract_ids`. A contract
       can gain consumers across separate ADRs that a later ADR's text never re-mentions —
       confirmed in production: `Catalog Listing` (`ibm` tenant) gained a consumer this way with
       no rename in that ADR's own text. The component's own payload does not change here, so
       it is persisted as-is with `force_new_version=True` — otherwise the hash matches the
       latest row and no-ops, leaving its source ADR pointing at the wrong one."""
    if not changed_contract_ids and not contract_directions_by_component_id:
        return
    live_components = await current_gold_state(session, "component", tenant=tenant)
    for row in live_components:
        if row.entity_id in mentioned_component_ids:
            continue
        payload = ComponentPayload.model_validate(row.payload)
        new_directions = contract_directions_by_component_id.get(row.entity_id, {"input": set(), "output": set()})
        already_linked = (set(payload.input_contract_ids) | set(payload.output_contract_ids)) & changed_contract_ids
        if not new_directions["input"] and not new_directions["output"] and not already_linked:
            continue

        if new_directions["input"] or new_directions["output"]:
            updated_payload = ComponentPayload(
                dependency_ids=payload.dependency_ids,
                contract_ids=payload.contract_ids,
                input_contract_ids=sorted(set(payload.input_contract_ids) | new_directions["input"]),
                output_contract_ids=sorted(set(payload.output_contract_ids) | new_directions["output"]),
            ).model_dump()
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=row.entity_id,
                canonical_name=row.canonical_name,
                operation="modified",
                narrative=row.narrative,
                payload=updated_payload,
                source_component=source_component,
                source_adr_version=source_adr_version,
                ingestion_date=ingestion_date,
                tenant=tenant,
                authored_by=row.authored_by,
            )
        else:
            await persist_entity_version(
                session,
                entity_type="component",
                entity_id=row.entity_id,
                canonical_name=row.canonical_name,
                operation="modified",
                narrative=row.narrative,
                payload=row.payload,
                source_component=source_component,
                source_adr_version=source_adr_version,
                ingestion_date=ingestion_date,
                tenant=tenant,
                authored_by=row.authored_by,
                force_new_version=True,
            )


async def extract_and_persist_gold_facts(
    session: AsyncSession,
    adr_content: str,
    source_component: str,
    source_adr_version: int,
    ingestion_date: date,
    *,
    tenant: str = "default",
    force: bool = False,
) -> bool:
    """Extracts Gold facts from one finished ADR and saves every resolved component, contract,
    and architecture entity. Shared body for two callers outside the LangGraph flow:
    `scripts/backfill_gold.py` and the frontend's "Publish" finalize endpoint
    (`app/routers/frontend.py`), which never run through the graph's own Gold nodes.

    Does not replace the three graph nodes (`extract_gold_facts`, `resolve_gold_identity`,
    `persist_gold_evolution` in `agents/graph.py`) — those stay as separately traced spans.

    Skips with no LLM call, returning `False`, if this exact `(tenant, source_component,
    source_adr_version)` was already extracted. `force=True` re-runs the LLM call anyway;
    `persist_entity_version` still no-ops any entity whose result is identical. Returns `True`
    only when it actually extracted.

    `pg_advisory_xact_lock` closes a real race: two overlapping calls for the same key (e.g. a
    double-clicked Publish button) can both read `already_extracted == False` before either
    commits, both call the LLM, and both persist — since two LLM calls rarely phrase a
    narrative identically, this produced a spurious v2 for every entity in practice. The lock
    serializes the check-extract-persist sequence per key; a second call blocks, then re-reads
    `already_extracted` as `True`. It is transaction-scoped and releases at commit/rollback."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:lock_key), :version)"),
        {"lock_key": f"{tenant}:{source_component}", "version": source_adr_version},
    )
    if not force and await already_extracted(session, source_component, source_adr_version, tenant=tenant):
        return False

    result = await extract_gold_facts_for_source(adr_content)
    # `adr_content` already has its `**Authors:** <username>` line by the time any caller
    # reaches here, so reading it back needs no extra parameter.
    authored_by = extract_authors_line(adr_content)
    name_to_id: dict[str, str] = {}

    # The prompt no longer asks for "unchanged"/"unknown" components or contracts, but
    # `GoldOperation` still allows those values — these skips are the backstop if a model
    # returns one anyway. `apply_contract_ripple_effect` covers a genuinely-affected component
    # this skip lets through un-mentioned.
    for component in result.components:
        if component.status in ("unknown", "unchanged"):
            continue
        await resolve_and_alias(
            session, "component", component.name, name_to_id, source_component, source_adr_version, tenant=tenant
        )
        for dep_name in component.dependency_names:
            await resolve_and_alias(
                session, "component", dep_name, name_to_id, source_component, source_adr_version, tenant=tenant
            )
        for contract_name in component.contract_names:
            await resolve_and_alias(
                session,
                "data_contract",
                contract_name,
                name_to_id,
                source_component,
                source_adr_version,
                tenant=tenant,
            )
    for contract in result.contracts:
        if contract.action in ("unknown", "unchanged"):
            continue
        await resolve_and_alias(
            session, "data_contract", contract.name, name_to_id, source_component, source_adr_version, tenant=tenant
        )
        for component_name in (contract.producer, contract.consumer):
            await resolve_and_alias(
                session, "component", component_name, name_to_id, source_component, source_adr_version, tenant=tenant
            )

    contract_dicts = [c.model_dump() for c in result.contracts]
    contract_directions = classify_contract_directions(contract_dicts, name_to_id)
    changed_contract_ids = contracts_with_real_changes(contract_dicts, name_to_id)
    # Re-keyed from component name to `entity_id` for `apply_contract_ripple_effect` below.
    contract_directions_by_component_id = {
        name_to_id[key]: directions
        for name, directions in contract_directions.items()
        if (key := memo_key("component", name)) in name_to_id
    }
    mentioned_component_ids: set[str] = set()
    for component in result.components:
        if component.status in ("unknown", "unchanged"):
            continue
        directions = contract_directions.get(component.name, {"input": set(), "output": set()})
        entity_id = name_to_id[memo_key("component", component.name)]
        mentioned_component_ids.add(entity_id)
        payload = ComponentPayload(
            dependency_ids=sorted(
                {name_to_id[key] for n in component.dependency_names if (key := memo_key("component", n)) in name_to_id}
            ),
            contract_ids=sorted(
                {name_to_id[key] for n in component.contract_names if (key := memo_key("data_contract", n)) in name_to_id}
            ),
            input_contract_ids=sorted(directions["input"]),
            output_contract_ids=sorted(directions["output"]),
        ).model_dump()
        await persist_entity_version(
            session,
            entity_type="component",
            entity_id=entity_id,
            canonical_name=component.name,
            operation=component.status,
            narrative=component.narrative,
            payload=payload,
            source_component=source_component,
            source_adr_version=source_adr_version,
            ingestion_date=ingestion_date,
            tenant=tenant,
            authored_by=authored_by,
        )
    for contract in result.contracts:
        if contract.action in ("unknown", "unchanged"):
            continue
        odcs_spec = parse_odcs_spec(contract.odcs_spec)
        if not odcs_spec:
            odcs_spec = await latest_odcs_spec(session, name_to_id[memo_key("data_contract", contract.name)], tenant)
        producer_id = name_to_id.get(memo_key("component", contract.producer), "")
        consumer_id = name_to_id.get(memo_key("component", contract.consumer), "")
        payload = DataContractPayload(
            producer=contract.producer,
            consumer=contract.consumer,
            producer_id=producer_id,
            consumer_id=consumer_id,
            odcs_spec=odcs_spec,
        ).model_dump()
        await persist_entity_version(
            session,
            entity_type="data_contract",
            entity_id=name_to_id[memo_key("data_contract", contract.name)],
            canonical_name=contract.name,
            operation=contract.action,
            narrative=contract.narrative,
            payload=payload,
            source_component=source_component,
            source_adr_version=source_adr_version,
            ingestion_date=ingestion_date,
            tenant=tenant,
            authored_by=authored_by,
        )

    await apply_contract_ripple_effect(
        session,
        changed_contract_ids=changed_contract_ids,
        contract_directions_by_component_id=contract_directions_by_component_id,
        mentioned_component_ids=mentioned_component_ids,
        source_component=source_component,
        source_adr_version=source_adr_version,
        ingestion_date=ingestion_date,
        tenant=tenant,
    )

    # `ArchitecturePayload` is a snapshot of the WHOLE architecture, not just this ADR's own
    # entries. Union them with every other live component's last-known data from
    # `gold_evolution`, so the LLM does not need to re-list what already existed untouched.
    live_components = [
        row for row in await current_gold_state(session, "component", tenant=tenant) if row.operation != "removed"
    ]
    id_to_name = {row.entity_id: row.canonical_name for row in live_components}
    all_component_names = {c.name for c in result.components if c.status != "removed"}
    all_dependency_names = {dep for c in result.components if c.status != "removed" for dep in c.dependency_names}
    for row in live_components:
        if row.entity_id in mentioned_component_ids:
            continue  # already folded in above
        all_component_names.add(row.canonical_name)
        payload = ComponentPayload.model_validate(row.payload)
        all_dependency_names.update(id_to_name.get(i, i) for i in payload.dependency_ids)

    architecture_payload = ArchitecturePayload(
        mermaid_diagram=result.mermaid_diagram,
        components=sorted(all_component_names),
        dependencies=sorted(all_dependency_names),
    ).model_dump()
    await persist_entity_version(
        session,
        entity_type="architecture",
        entity_id=f"architecture:{source_component}",
        canonical_name=source_component,
        operation=result.architecture_change,
        narrative=result.architecture_narrative,
        payload=architecture_payload,
        source_component=source_component,
        source_adr_version=source_adr_version,
        ingestion_date=ingestion_date,
        tenant=tenant,
        authored_by=authored_by,
    )
    return True


async def current_gold_state(
    session: AsyncSession, entity_type: GoldEntityType | None = None, *, tenant: str = "default"
) -> Sequence[GoldEvolution]:
    """Current-state read: one row per `(entity_type, entity_id)`, always the latest version,
    queried live over `gold_evolution` instead of a cache.

    Always scopes to one `tenant`. Used by the Chat tab and `current_architecture_diagram` —
    neither should ever see another tenant's architecture."""
    query = select(GoldEvolution).distinct(GoldEvolution.entity_type, GoldEvolution.entity_id).where(
        GoldEvolution.tenant == tenant
    )
    if entity_type is not None:
        query = query.where(GoldEvolution.entity_type == entity_type)
    query = query.order_by(GoldEvolution.entity_type, GoldEvolution.entity_id, GoldEvolution.version.desc())
    return (await session.execute(query)).scalars().all()


async def current_gold_state_as_of(
    session: AsyncSession, as_of: date, entity_type: GoldEntityType | None = None, *, tenant: str = "default"
) -> Sequence[GoldEvolution]:
    """Same read path as `current_gold_state`, pinned to whatever was valid on `as_of` instead
    of the latest version. `valid_to` makes each entity's version windows non-overlapping, so
    at most one row per `(entity_type, entity_id)` matches."""
    query = (
        select(GoldEvolution)
        .distinct(GoldEvolution.entity_type, GoldEvolution.entity_id)
        .where(
            GoldEvolution.tenant == tenant,
            GoldEvolution.ingestion_date <= as_of,
            or_(GoldEvolution.valid_to.is_(None), GoldEvolution.valid_to > as_of),
        )
    )
    if entity_type is not None:
        query = query.where(GoldEvolution.entity_type == entity_type)
    query = query.order_by(GoldEvolution.entity_type, GoldEvolution.entity_id, GoldEvolution.version.desc())
    return (await session.execute(query)).scalars().all()


async def _batch_cursor(session: AsyncSession, tenant: str, row: GoldEvolution) -> tuple[datetime, int]:
    """The `(changed_at, id)` of the LAST row inserted in the same extraction batch as `row` —
    same `(tenant, source_component, source_adr_version)` — not `row`'s own values.

    One ADR persists several entities as separate, sequential inserts, so a component's own
    row can have a lower id than a contract from the same ADR. Cursoring on the component's own
    id wrongly excluded that sibling as "not yet known" — a real bug: a component with no
    dependencies of its own, related only through a same-ADR sibling, showed no relationship at
    all. The batch's last id includes every fact from that ADR regardless of write order."""
    last = (
        await session.execute(
            select(GoldEvolution.changed_at, GoldEvolution.id)
            .where(
                GoldEvolution.tenant == tenant,
                GoldEvolution.source_component == row.source_component,
                GoldEvolution.source_adr_version == row.source_adr_version,
            )
            .order_by(GoldEvolution.id.desc())
            .limit(1)
        )
    ).one()
    return last.changed_at, last.id


async def _rows_before(
    session: AsyncSession,
    entity_type: GoldEntityType,
    tenant: str,
    before: tuple[datetime, int],
    entity_ids: Sequence[str] | None = None,
) -> list[GoldEvolution]:
    """The latest row per entity among rows inserted at or before `before` — a `(changed_at,
    id)` cursor, never `ingestion_date` alone: several versions of the same entity can share
    one `ingestion_date`, so date-only comparison could not tell which same-day version came
    first — a real bug (a v1-pinned diagram still showed a v3-only dependency).

    Used only to resolve a SPECIFIC row's own neighbors/successors (`build_relationship_diagram`,
    `build_context_lines`) — not for an arbitrary business date, which stays
    `current_gold_state_as_of`/`_entities_as_of`'s `as_of`."""
    changed_at, entity_pk = before
    query = select(GoldEvolution).distinct(GoldEvolution.entity_id).where(
        GoldEvolution.tenant == tenant,
        GoldEvolution.entity_type == entity_type,
        tuple_(GoldEvolution.changed_at, GoldEvolution.id) <= tuple_(changed_at, entity_pk),
    )
    if entity_ids is not None:
        if not entity_ids:
            return []
        query = query.where(GoldEvolution.entity_id.in_(entity_ids))
    query = query.order_by(GoldEvolution.entity_id, GoldEvolution.changed_at.desc(), GoldEvolution.id.desc())
    return (await session.execute(query)).scalars().all()


async def _entities_as_of(
    session: AsyncSession,
    entity_type: GoldEntityType,
    entity_ids: Sequence[str],
    tenant: str,
    as_of: date | None,
    *,
    before: tuple[datetime, int] | None = None,
) -> list[GoldEvolution]:
    """Batched `entity_id -> its row valid at as_of` lookup, one query for the whole list,
    scoped to a specific set of ids. `as_of=None` means the latest version.

    `before`, when given, takes over entirely (see `_rows_before`) — for resolving one specific
    row's own neighbors, not a business date."""
    if not entity_ids:
        return []
    if before is not None:
        return await _rows_before(session, entity_type, tenant, before, entity_ids)
    query = select(GoldEvolution).distinct(GoldEvolution.entity_id).where(
        GoldEvolution.tenant == tenant,
        GoldEvolution.entity_type == entity_type,
        GoldEvolution.entity_id.in_(entity_ids),
    )
    if as_of is not None:
        query = query.where(
            GoldEvolution.ingestion_date <= as_of,
            or_(GoldEvolution.valid_to.is_(None), GoldEvolution.valid_to > as_of),
        )
    query = query.order_by(GoldEvolution.entity_id, GoldEvolution.version.desc())
    return (await session.execute(query)).scalars().all()


async def get_component_contracts(
    session: AsyncSession, component_id: str, tenant: str = "default", as_of: date | None = None
) -> dict[str, list[DataContractPayload]]:
    """The full ODCS spec (`DataContractPayload`, not just names) of a component's data
    contracts, split by direction. Reads each spec fresh from the contract's own row valid at
    `as_of`, never a copy stored on the component itself."""
    components = await _entities_as_of(session, "component", [component_id], tenant, as_of)
    if not components:
        return {"input": [], "output": []}
    payload = ComponentPayload.model_validate(components[0].payload)
    input_rows = await _entities_as_of(session, "data_contract", payload.input_contract_ids, tenant, as_of)
    output_rows = await _entities_as_of(session, "data_contract", payload.output_contract_ids, tenant, as_of)
    return {
        "input": [DataContractPayload.model_validate(r.payload) for r in input_rows],
        "output": [DataContractPayload.model_validate(r.payload) for r in output_rows],
    }


async def get_predecessors(
    session: AsyncSession, component_id: str, tenant: str = "default", as_of: date | None = None
) -> list[GoldEvolution]:
    """Components this one directly depends on (`ComponentPayload.dependency_ids`), resolved to
    each one's own row valid at `as_of`. A direct read — see `get_successors` for the inverse."""
    components = await _entities_as_of(session, "component", [component_id], tenant, as_of)
    if not components:
        return []
    payload = ComponentPayload.model_validate(components[0].payload)
    return await _entities_as_of(session, "component", payload.dependency_ids, tenant, as_of)


async def get_successors(
    session: AsyncSession,
    component_id: str,
    tenant: str = "default",
    as_of: date | None = None,
    *,
    before: tuple[datetime, int] | None = None,
) -> list[GoldEvolution]:
    """The inverse of `get_predecessors` — every component whose own `dependency_ids` names
    `component_id`, as of `as_of` (a business date) or `before` (a `(changed_at, id)` cursor).
    There is no `successor_ids` field: this is derived from `dependency_ids` so it can never
    drift out of sync with it."""
    if before is not None:
        candidates = await _rows_before(session, "component", tenant, before)
    else:
        candidates = (
            await current_gold_state_as_of(session, as_of, "component", tenant=tenant)
            if as_of is not None
            else await current_gold_state(session, "component", tenant=tenant)
        )
    return [
        row for row in candidates if component_id in ComponentPayload.model_validate(row.payload).dependency_ids
    ]


async def current_architecture_diagram(session: AsyncSession, *, tenant: str = "default") -> str:
    """Builds a Mermaid `flowchart LR` from Gold's own current component state. NOT the
    LLM-drawn diagram on one ADR's `entity_type="architecture"` row
    (`ArchitecturePayload.mermaid_diagram`), which only reflects that one ADR. Reads
    `current_gold_state`'s live view instead, across every ADR, with `removed` components
    dropped.

    Each node's label names the component's current Gold `version` (e.g. "Backend v4"). Returns
    `""` if Gold has no live component yet.

    The label used to also carry a subtitle naming the last ADR that touched each node. Dropped:
    this diagram feeds an LLM as grounding context (`generate_architecture_questions`,
    `synthesize_document`), and `synthesize_document` passes it as "PREVIOUS_ARCHITECTURE_DIAGRAM"
    for the model to reproduce verbatim — so the subtitle leaked into every newly generated ADR
    permanently. "View ADR" links in the table below the diagram handle that navigation instead.

    No `click <id> href ...` directive on nodes. ROOT CAUSE (confirmed against a running
    `st.mermaid_chart`): that directive makes `st.mermaid_chart` render zero node elements,
    only edges — `mermaid-cli` does not reproduce this. Do not add `click` back without
    re-verifying against real `st.mermaid_chart`, not just `mermaid-cli`.

    Node labels use Mermaid's backtick markdown-string syntax (not raw HTML) and an explicit
    `classDef`/`class` for color, never the default theme fill.

    Edges are `component --> dependency`, from `ComponentPayload.dependency_ids`. A dependency
    that is not itself a live component is silently skipped, not drawn as a dangling node."""
    components = await current_gold_state(session, "component", tenant=tenant)
    live = [c for c in components if c.operation != "removed"]
    if not live:
        return ""

    node_id = {c.entity_id: f"n{i}" for i, c in enumerate(live)}
    lines = [
        "flowchart LR",
        "    classDef goldNode fill:#f1f3f5,stroke:#8b5cf6,color:#16181d,stroke-width:1px",
    ]
    for c in live:
        lines.append(f'    {node_id[c.entity_id]}["`{c.canonical_name} v{c.version}`"]')
    for c in live:
        dependency_ids = ComponentPayload.model_validate(c.payload).dependency_ids
        for dependency_id in dependency_ids:
            if dependency_id in node_id and dependency_id != c.entity_id:
                lines.append(f"    {node_id[c.entity_id]} --> {node_id[dependency_id]}")
    lines.append(f"    class {','.join(node_id.values())} goldNode")
    return "\n".join(lines)


async def current_architecture_diagram_interactive(
    session: AsyncSession, *, tenant: str = "default"
) -> tuple[str, dict[str, str]]:
    """Builds a Mermaid `flowchart LR` of Gold's current component state for
    `streamlit_mermaid_interactive` (a third-party component, not `st.mermaid_chart`), to render
    with real click events.

    `streamlit_mermaid_interactive` renders into the page's live DOM with its own JS click
    listeners, never Mermaid's `click` directive, so it avoids the rendering bug described in
    `current_architecture_diagram`'s docstring.

    Returns `(mermaid_code, entity_name_mapping)`. `entity_name_mapping` maps a node's visible
    label text to `"{source_component}::{source_adr_version}"`, because this component
    identifies the clicked node by its rendered text, not an internal node id — the caller reads
    this back to build the `?view_adr=...&view_adr_version=...` navigation. Each label is a
    single line (no italic subtitle), since a multi-line label's rendered text cannot be
    verified without a real browser. Returns `("", {})` if Gold has no live component yet."""
    components = await current_gold_state(session, "component", tenant=tenant)
    live = [c for c in components if c.operation != "removed"]
    if not live:
        return "", {}

    by_id = {c.entity_id: c for c in live}
    node_id = {c.entity_id: f"n{i}" for i, c in enumerate(live)}
    lines = [
        "flowchart LR",
        "    classDef goldNode fill:#f1f3f5,stroke:#8b5cf6,color:#16181d,stroke-width:1px",
    ]
    mapping: dict[str, str] = {}
    for c in live:
        label = f"{c.canonical_name} v{c.version}"
        lines.append(f'    {node_id[c.entity_id]}["`{label}`"]')
        mapping[label] = f"{c.source_component}::{c.source_adr_version}"
    for c in live:
        for dependency_id in ComponentPayload.model_validate(c.payload).dependency_ids:
            if dependency_id in by_id and dependency_id != c.entity_id:
                lines.append(f"    {node_id[c.entity_id]} --> {node_id[dependency_id]}")
    lines.append(f"    class {','.join(node_id.values())} goldNode")
    return "\n".join(lines), mapping


async def build_relationship_diagram(
    session: AsyncSession, rows: list[GoldEvolution], *, tenant: str = "default"
) -> tuple[str, list[GoldEvolution]] | None:
    """A small Mermaid `flowchart LR` scoped to just the component(s) a chat answer cited
    (`rows`), plus each one's direct neighbors: predecessors (`dependency_ids`) and successors
    (`get_successors`). Built from the exact rows an answer resolved, so it never shows a
    component the answer did not actually ground.

    Returns `None` when there is nothing to show. Otherwise returns `(mermaid_source,
    focal_rows)` — `focal_rows` is exactly the cited component rows, so the caller can build
    "View ADR" links from the entity the answer was about.

    A historical focal row (`valid_to is not None`) resolves its neighbors with a cursor pinned
    to that row's own insertion moment, never today's state. Fixed a real bug: an answer pinned
    to an old version still showed a dependency added long after it, because `get_successors`
    was called unscoped. `ingestion_date` alone cannot fix this since several versions can share
    one calendar day. A still-current focal row keeps reading today's true state.

    Reuses `current_architecture_diagram`'s style conventions. A second `classDef`
    (`focalNode`) highlights the cited component(s)."""
    focal_rows = [row for row in rows if row.entity_type == "component"]
    if not focal_rows:
        return None

    focal_ids = {row.entity_id for row in focal_rows}
    # (from_id, to_id): from_id depends on to_id, same direction as component --> dependency.
    edges: list[tuple[str, str]] = []
    neighbor_rows_by_id: dict[str, GoldEvolution] = {}
    for row in focal_rows:
        cursor = await _batch_cursor(session, tenant, row) if row.valid_to is not None else None
        payload = ComponentPayload.model_validate(row.payload or {})
        for dependency_id in payload.dependency_ids:
            edges.append((row.entity_id, dependency_id))
        to_fetch = [i for i in payload.dependency_ids if i not in focal_ids]
        if to_fetch:
            for neighbor in await _entities_as_of(session, "component", to_fetch, tenant, None, before=cursor):
                neighbor_rows_by_id[neighbor.entity_id] = neighbor
        for successor in await get_successors(session, row.entity_id, tenant=tenant, before=cursor):
            edges.append((successor.entity_id, row.entity_id))
            if successor.entity_id not in focal_ids:
                neighbor_rows_by_id[successor.entity_id] = successor

    if not edges:
        return None  # every focal component has zero dependencies and zero successors

    all_rows = {**neighbor_rows_by_id, **{row.entity_id: row for row in focal_rows}}

    node_id = {entity_id: f"n{i}" for i, entity_id in enumerate(all_rows)}
    lines = [
        "flowchart LR",
        "    classDef goldNode fill:#f1f3f5,stroke:#8b5cf6,color:#16181d,stroke-width:1px",
        "    classDef focalNode fill:#ede9fe,stroke:#7c3aed,color:#16181d,stroke-width:2px",
    ]
    for entity_id, row in all_rows.items():
        lines.append(f'    {node_id[entity_id]}["`{row.canonical_name}`"]')
    seen_edges: set[tuple[str, str]] = set()
    for from_id, to_id in edges:
        if from_id not in node_id or to_id not in node_id or from_id == to_id:
            continue
        edge = (node_id[from_id], node_id[to_id])
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        lines.append(f"    {edge[0]} --> {edge[1]}")
    neighbor_node_ids = [node_id[i] for i in all_rows if i not in focal_ids]
    focal_node_ids = [node_id[i] for i in focal_ids if i in node_id]
    if neighbor_node_ids:
        lines.append(f"    class {','.join(neighbor_node_ids)} goldNode")
    if focal_node_ids:
        lines.append(f"    class {','.join(focal_node_ids)} focalNode")
    return "\n".join(lines), focal_rows


# --- Top-k retrieval: the RAG read path. `current_gold_state` answers "latest version per
# entity"; this answers "which versions are semantically closest to a question." Shared by
# `scripts/chat_gold.py` and `agents/stages/gold/testing/retrieval.py`.

# Cosine distance above which a row is dropped from top-k, instead of forcing exactly `k` rows
# regardless of relevance. Not yet tuned against a real corpus — a starting point.
#
# Opt-in on `top_k_gold_evolution`, default `None` (unfiltered), so tests are not silently
# affected by changing this default.
DEFAULT_MAX_DISTANCE = 0.6

# Candidates fetched before cutting down to `k`. Three things need this same extra depth:
#
# 1. `mode="hybrid"` fuses vector and lexical rankings with RRF — fusing lists already cut to
#    `k` would never let a moderate-vector/strong-lexical row win.
# 2. `dedupe=True` collapses several versions of the same entity to one before cutting to `k`,
#    so near-identical versions don't crowd out a different relevant entity.
# 3. `rerank=True` repunctuates the pool with a cross-encoder before cutting to `k` — reranking
#    a list already cut to `k` could never promote a candidate ranked just outside it.
RECALL_POOL_SIZE = 30

# `RRF_K_CONSTANT`, `_reciprocal_rank_fusion`, `_top_k_ids_by_vector`, `_top_k_ids_by_lexical_rank`
# (hybrid search), `_dedupe_ids_by_entity` (deduplication), and `_rerank_ids` (reranking) live in
# `agents/stages/gold/retrieval/`. This function orchestrates them.


async def embed_question(text: str) -> list[float]:
    """Wraps the blocking, batched `ingestion.embedder.embed` in `asyncio.to_thread`, for one
    question at a time."""
    [vector] = await asyncio.to_thread(embed, [text])
    return vector


async def top_k_gold_evolution(
    session: AsyncSession,
    vector: list[float],
    k: int = 5,
    source_component: str | None = None,
    max_distance: float | None = None,
    *,
    tenant: str = "default",
    mode: Literal["vector", "hybrid"] = "vector",
    question_text: str | None = None,
    dedupe: bool = True,
    rerank: bool = False,
    expand: bool = False,
) -> list[GoldEvolution]:
    """Top-k search over ALL versions of `gold_evolution`, not just the latest per entity
    (that is `current_gold_state`'s job).

    Always scopes to one `tenant` first. `source_component`, when given, narrows to one source
    within that tenant.

    `max_distance` (see `DEFAULT_MAX_DISTANCE`) drops rows past that distance in SQL, instead of
    always forcing `k` rows. Applies only to the vector ranking — `ts_rank`'s scale is not
    comparable to cosine distance.

    `mode="vector"` (default) orders by `.cosine_distance(vector)`, matching the HNSW index.
    `mode="hybrid"` also runs a lexical `ts_rank` search and fuses both rankings with
    `_reciprocal_rank_fusion`: an embedding can blur an exact name or acronym that a keyword
    match finds immediately. Requires `question_text` — the lexical branch tokenizes the raw
    question itself, not `vector`.

    `dedupe=True` (default) collapses several versions of the same entity to one, its latest —
    see `_dedupe_ids_by_entity`. `dedupe=False` reproduces the plain pre-deduplication top-`k`
    cut, kept for tests needing the raw ranking.

    `rerank=True` (opt-in) repunctuates the pool with a cross-encoder
    (`ingestion.reranker.score_candidates`) before cutting to `k` — see `_rerank_ids`. Requires
    `question_text` in every mode, since the cross-encoder needs the raw question text.

    `expand=True` (opt-in) asks `expand_question` for reformulations of `question_text`,
    searches with each, and fuses those rankings into the same pool — see
    `retrieval/query_expansion.py`. Closes the gap hybrid search alone cannot: a synonym sharing
    no words with the canonical name may not match lexically or embed close enough either.
    Requires `question_text`, same reason as `rerank=True`."""
    if rerank and not question_text:
        raise ValueError("rerank=True requires question_text (the cross-encoder needs the raw question)")
    if expand and not question_text:
        raise ValueError("expand=True requires question_text (needed to generate reformulations)")

    # Hybrid search and query expansion both fuse several rankings, so both need the same wide
    # recall depth as dedup/rerank — a shallow fetch would starve the fusion of candidates.
    fuse_deep = mode == "hybrid" or expand
    recall = RECALL_POOL_SIZE if (dedupe or rerank or fuse_deep) else k

    rankings: list[list[int]] = []
    if mode == "vector":
        rankings.append(await _top_k_ids_by_vector(session, vector, recall, source_component, max_distance, tenant))
    elif mode == "hybrid":
        if not question_text:
            raise ValueError("mode='hybrid' requires question_text (the lexical ranking needs the raw question)")
        # Sequential, not `asyncio.gather` — one `AsyncSession` cannot run two queries
        # concurrently. `gather` here raised a real `sqlalchemy.exc.IllegalStateChangeError`.
        rankings.append(await _top_k_ids_by_vector(session, vector, recall, source_component, max_distance, tenant))
        rankings.append(await _top_k_ids_by_lexical_rank(session, question_text, recall, source_component, tenant))
    else:
        raise ValueError(f"unknown top_k_gold_evolution mode: {mode!r}")

    if expand:
        for variant_text in await expand_question(question_text):
            variant_vector = await embed_question(variant_text)
            rankings.append(
                await _top_k_ids_by_vector(session, variant_vector, recall, source_component, max_distance, tenant)
            )
            if mode == "hybrid":
                rankings.append(await _top_k_ids_by_lexical_rank(session, variant_text, recall, source_component, tenant))

    ids = rankings[0] if len(rankings) == 1 else _reciprocal_rank_fusion(rankings)

    # `rerank=True` needs the deduped pool BEFORE it is cut to `k`, so dedup keeps up to
    # `RECALL_POOL_SIZE` entities here whenever a rerank pass still follows.
    dedupe_limit = RECALL_POOL_SIZE if rerank else k
    ids = await _dedupe_ids_by_entity(session, ids, dedupe_limit) if dedupe else ids[:dedupe_limit]

    ids = await _rerank_ids(session, question_text, ids, k) if rerank else ids[:k]

    if not ids:
        return []
    rows = (await session.execute(select(GoldEvolution).where(GoldEvolution.id.in_(ids)))).scalars().all()
    rows_by_id = {row.id: row for row in rows}
    return [rows_by_id[item_id] for item_id in ids if item_id in rows_by_id]


async def latest_versions(session: AsyncSession, rows: Sequence[GoldEvolution]) -> dict[tuple[str, str], int]:
    """The current latest version number for each distinct `(entity_type, entity_id)` in
    `rows`, in one batched query.

    Lets `answer_question` tell the LLM which retrieved rows are current and which are
    superseded, since top-k search runs across all versions and an older one can rank closer
    to the question than the current version."""
    pairs = {(row.entity_type, row.entity_id) for row in rows}
    if not pairs:
        return {}
    query = (
        select(GoldEvolution.entity_type, GoldEvolution.entity_id, func.max(GoldEvolution.version))
        .where(or_(*(and_(GoldEvolution.entity_type == et, GoldEvolution.entity_id == eid) for et, eid in pairs)))
        .group_by(GoldEvolution.entity_type, GoldEvolution.entity_id)
    )
    result = await session.execute(query)
    return {(entity_type, entity_id): version for entity_type, entity_id, version in result.all()}


def _version_tag(row: GoldEvolution, latest: dict[tuple[str, str], int] | None) -> str:
    if latest is None:
        return ""
    latest_version = latest.get((row.entity_type, row.entity_id))
    if latest_version is None or latest_version == row.version:
        return " [latest version]"
    return f" [superseded — latest is version {latest_version}]"


async def _resolve_entity_names(
    session: AsyncSession, tenant: str, ids_by_type: dict[str, set[str]]
) -> dict[tuple[str, str], str]:
    """Batched `entity_id -> canonical_name` lookup for `answer_question`'s payload
    enrichment. Payload fields only store resolved ids, never names, and a raw ULID is useless
    to the answering LLM. One query per entity_type, not one per id."""
    result: dict[tuple[str, str], str] = {}
    for entity_type, ids in ids_by_type.items():
        if not ids:
            continue
        rows = (
            await session.execute(
                select(GoldEvolution.entity_id, GoldEvolution.canonical_name, GoldEvolution.version)
                .where(
                    GoldEvolution.tenant == tenant,
                    GoldEvolution.entity_type == entity_type,
                    GoldEvolution.entity_id.in_(ids),
                )
                .order_by(GoldEvolution.entity_id, GoldEvolution.version.desc())
            )
        ).all()
        seen: set[str] = set()
        for entity_id, canonical_name, _version in rows:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            result[(entity_type, entity_id)] = canonical_name
    return result


def _payload_detail(
    row: GoldEvolution, names: dict[tuple[str, str], str], successors: dict[str, list[str]] | None = None
) -> str:
    """Renders `row.payload`'s structured facts as a short trailing clause for
    `answer_question`'s context line — the detail a dependency/contract question needs that
    `row.narrative` alone does not reliably restate. Returns `""` when the payload has nothing
    to add.

    `successors`, when given, maps a component's `entity_id` to the names of every other
    component that currently depends on it (`get_successors`'s result, reduced to names by the
    caller) — there is no stored `successor_ids` field; it is always computed."""
    if row.entity_type == "component":
        payload = ComponentPayload.model_validate(row.payload or {})
        deps = [names[("component", i)] for i in payload.dependency_ids if ("component", i) in names]
        parts = []
        if deps:
            parts.append(f"depends on: {', '.join(deps)}")
        succ = (successors or {}).get(row.entity_id, [])
        if succ:
            parts.append(f"depended on by: {', '.join(succ)}")
        inputs = [names[("data_contract", i)] for i in payload.input_contract_ids if ("data_contract", i) in names]
        outputs = [names[("data_contract", i)] for i in payload.output_contract_ids if ("data_contract", i) in names]
        if inputs:
            parts.append(f"input contracts: {', '.join(inputs)}")
        if outputs:
            parts.append(f"output contracts: {', '.join(outputs)}")
        if not inputs and not outputs:
            # Rows persisted before the input/output split only have the old, undifferentiated
            # `contract_ids` — fall back to it so they do not go silent.
            contracts = [names[("data_contract", i)] for i in payload.contract_ids if ("data_contract", i) in names]
            if contracts:
                parts.append(f"data contracts: {', '.join(contracts)}")
        return f" ({'; '.join(parts)})" if parts else ""
    if row.entity_type == "data_contract":
        payload = DataContractPayload.model_validate(row.payload or {})
        parts = []
        if payload.producer or payload.consumer:
            parts.append(f"producer: {payload.producer or 'unknown'}, consumer: {payload.consumer or 'unknown'}")
        fields = list(payload.odcs_spec.get("schema", {}).get("properties", {}).keys())
        if fields:
            parts.append(f"schema fields: {', '.join(fields)}")
        return f" ({'; '.join(parts)})" if parts else ""
    return ""


# `MAX_HISTORY_MESSAGES`/`_format_history_block` (conversational memory) and `_verify_citations`
# (citation verification) now live in `agents/stages/gold/retrieval/`, imported above.

# Fixes a real bug: `_payload_detail`'s data_contract branch only showed schema field names,
# never the actual spec, so "show me the spec of X" wrongly said it was unavailable. Gates the
# full spec block behind an explicit ask, so an ordinary question does not pay for it.
_FULL_SPEC_MARKERS = (
    "spec",
    "specification",
    "schema",
    "definition",
    "yaml",
    "full detail",
    "especificación",
    "especificacion",
    "esquema",
    "definición",
    "definicion",
)


def wants_full_spec(question: str) -> bool:
    """Returns `True` if `question` asks for a data contract's complete specification, in
    English or Spanish — see `_FULL_SPEC_MARKERS`. Gates `build_context_lines`'s full-spec
    block."""
    lowered = question.lower()
    return any(marker in lowered for marker in _FULL_SPEC_MARKERS)


def _full_odcs_spec_block(row: GoldEvolution) -> str:
    """The entire `odcs_spec` for one data_contract row, pretty-printed. `""` for any other
    entity_type, or a contract with no spec captured yet."""
    if row.entity_type != "data_contract":
        return ""
    payload = DataContractPayload.model_validate(row.payload or {})
    if not payload.odcs_spec:
        return ""
    return f"\n  Full specification: {json.dumps(payload.odcs_spec, indent=2)}"


async def build_context_lines(
    session: AsyncSession,
    rows: list[GoldEvolution],
    latest: dict[tuple[str, str], int] | None = None,
    question: str | None = None,
) -> list[str]:
    """One line per retrieved row, exactly as `answer_question` renders it into its prompt:
    narrative plus `_payload_detail`, plus a contract's full `odcs_spec` when `question` asks
    for it (`wants_full_spec`). Pulled out so a caller other than `answer_question` (RAGAS
    scoring in `test_chat_eval.py`) can score against what the model actually saw, not just
    the bare narrative.

    `question` is optional; `None` skips the full-spec block."""
    full_spec = question is not None and wants_full_spec(question)
    component_payloads = [
        ComponentPayload.model_validate(row.payload or {}) for row in rows if row.entity_type == "component"
    ]
    component_ids = {i for p in component_payloads for i in p.dependency_ids}
    contract_ids = {
        i for p in component_payloads for i in (p.contract_ids + p.input_contract_ids + p.output_contract_ids)
    }
    names = await _resolve_entity_names(
        session, rows[0].tenant, {"component": component_ids, "data_contract": contract_ids}
    )
    # One `get_successors` query per retrieved component — cheap since `rows` is a bounded
    # top-k. There is no stored successors field, so this is the only way to surface them.
    #
    # A historical row (`valid_to is not None`) is cursored to its own batch's end
    # (`_batch_cursor`), never "current" — leaving this unscoped was a real bug: a v1-pinned
    # answer listed a dependency added long after v1. A still-current row keeps reading today's
    # true state.
    successors_by_component_id: dict[str, list[str]] = {}
    for row in rows:
        if row.entity_type != "component":
            continue
        cursor = await _batch_cursor(session, row.tenant, row) if row.valid_to is not None else None
        successors = await get_successors(session, row.entity_id, tenant=row.tenant, before=cursor)
        successors_by_component_id[row.entity_id] = [s.canonical_name for s in successors]
    return [
        f"- [{row.entity_type}:{row.entity_id}] {row.canonical_name} (version {row.version}, "
        f"operation={row.operation}, {row.ingestion_date.isoformat()}, by "
        f"{row.authored_by or 'an unknown author'}, from ADR {row.source_component} "
        f"v{row.source_adr_version}){_version_tag(row, latest)}: {row.narrative}"
        f"{_payload_detail(row, names, successors_by_component_id)}"
        f"{_full_odcs_spec_block(row) if full_spec else ''}"
        for row in rows
    ]


async def answer_question(
    session: AsyncSession,
    question: str,
    rows: list[GoldEvolution],
    latest: dict[tuple[str, str], int] | None = None,
    history: list[tuple[str, str]] | None = None,
) -> GroundedAnswer:
    """Drafts a structured answer to `question` from the retrieved rows — plain-text `answer`
    plus `citations` naming exactly which rows it drew from.

    Every citation is checked against `rows` by `_verify_citations` before this returns: the
    model cannot make a citation stick unless it names a row that was genuinely retrieved.

    `latest`, from `latest_versions`, tags each row as `[latest version]` or
    `[superseded — latest is version N]`. This stops a current-state question from being
    answered using a row that ranked close by embedding similarity but is now superseded.

    `session` resolves each row's payload ids into names the LLM can read (`_payload_detail`);
    `rows` themselves are not re-fetched.

    `history`, when given, is the most recent turns of this chat session, rendered as its own
    labeled block, never merged into `question`. This lets a follow-up like "and who approved
    it?" resolve "it" to the right entity. History may only resolve what the question refers
    to, never serve as a source of facts — every fact must still come from `rows`."""
    if not rows:
        return GroundedAnswer(answer="No relevant Gold facts were found for this question.")

    context_lines = await build_context_lines(session, rows, latest, question)
    context = "\n\n".join(context_lines)
    messages = [
        {
            "role": "user",
            "content": (
                "Answer the question below using ONLY the retrieved architecture facts as "
                "context — do not invent anything the facts don't state. Default to concise "
                "(2-4 sentences); only go longer, covering every relevant fact in the context "
                "(including the parenthetical dependency/contract/schema detail after a fact, "
                "when present), when the question itself asks for detail, specifics, or a full "
                "picture (e.g. \"details\", \"detalles\", \"tell me everything about\", "
                "\"explain fully\") — length must track what was actually asked, never padded "
                "beyond what the facts support. If the facts describe an entity's evolution "
                "across versions, state its current/latest status explicitly, preferring facts "
                "tagged [latest version] for that; facts tagged [superseded] describe history, "
                "not the current state, and should only be used to answer questions about how "
                "something evolved over time. Each fact's parenthetical also carries its "
                "ingestion date and author — use them directly for a \"when\"/\"who\" question; "
                "if the author shown is \"an unknown author\", say the author is not recorded "
                "instead of claiming no information exists at all. Each fact also names the "
                "exact ADR it came from (\"from ADR <source_component> v<version>\") — give this "
                "back verbatim, source_component and version both, whenever the question asks "
                "which ADR/document a fact comes from, so the reader can look that exact ADR up "
                "in the Architecture history table. If a previous conversation is given below, use it "
                "ONLY to resolve what the question refers to (a pronoun, \"it\", \"that "
                "component\") — never as a source of facts on its own; every fact in your "
                "answer must still come from the retrieved facts.\n\n"
                "Return a JSON object with two fields: \"answer\" (the text answer, as "
                "described above) and \"citations\" (a list of every fact you actually used, "
                "each as {\"entity_type\": ..., \"entity_id\": ..., \"version\": ...} — copy "
                "these three values verbatim from the \"[entity_type:entity_id]\" tag and "
                "\"version\" of each fact you cite, never invent or guess one). A fact you did "
                "not end up using in the answer does not belong in citations.\n\n"
                f"{_format_history_block(history)}"
                f"Retrieved facts:\n{context}\n\nQuestion: {question}"
            ),
        }
    ]
    response = await router.complete(messages, response_format=GroundedAnswer, temperature=0, reasoning_effort="none")
    result = GroundedAnswer.model_validate(load_json_response(response.choices[0].message.content))
    return GroundedAnswer(answer=result.answer, citations=_verify_citations(result.citations, rows))


# --- Entity evolution timeline: a third read path, next to `current_gold_state` and
# `top_k_gold_evolution`. "How has X evolved" wants EVERY version of one identified entity, in
# order — top-k can drop an early version whose narrative does not word-match the question.
# This path fetches deterministically: find the named entity, take its full history.

# `find_entity_by_name_in_text` ignores any alias shorter than this — a short alias (an
# acronym, a word fragment) is too likely to match a question by coincidence.
MIN_ALIAS_MATCH_LENGTH = 4

# `is_evolution_question` triggers on any of these, case-insensitive, English and Spanish — a
# narrow deterministic check, not an LLM call. A caller only takes the full-history path when
# this ALSO holds alongside a confident `find_entity_by_name_in_text` match — an ordinary
# question naming a component must still get plain top-k retrieval.
_EVOLUTION_QUESTION_MARKERS = (
    "evolution",
    "evolved",
    "over time",
    "history",
    "historical",
    "timeline",
    "evolución",
    "evolucionado",
    "a lo largo del tiempo",
    "historia",
    "histórico",
    "histórica",
)


def is_evolution_question(question: str) -> bool:
    """Returns `True` if `question` asks for an entity's full history, English or Spanish —
    see `_EVOLUTION_QUESTION_MARKERS`. Gates the full-history path alongside a confident
    `find_entity_by_name_in_text` match, never on its own: naming an entity is not enough."""
    lowered = question.lower()
    return any(marker in lowered for marker in _EVOLUTION_QUESTION_MARKERS)


async def entity_history(
    session: AsyncSession, entity_type: GoldEntityType, entity_id: str, *, tenant: str = "default"
) -> list[GoldEvolution]:
    """Every `gold_evolution` row ever written for this `(entity_type, entity_id)`, oldest
    first. Deterministic counterpart to `top_k_gold_evolution`: no ranking, no cutoff. Feeds
    `answer_evolution_question`, which needs the complete timeline."""
    query = (
        select(GoldEvolution)
        .where(
            GoldEvolution.tenant == tenant,
            GoldEvolution.entity_type == entity_type,
            GoldEvolution.entity_id == entity_id,
        )
        .order_by(GoldEvolution.version.asc())
    )
    return list((await session.execute(query)).scalars().all())


async def find_entity_by_name_in_text(
    session: AsyncSession, text: str, *, tenant: str = "default", entity_type: GoldEntityType | None = None
) -> tuple[GoldEntityType, str, str] | None:
    """Looks for a known component or data-contract name inside free-form text, typically a
    chat question, and resolves it to a live `entity_id`. Returns `(entity_type, entity_id,
    matched_alias)`, or `None` if nothing in `gold_aliases` for this tenant appears in `text`.

    Checks containment in Python, not a SQL `LIKE` pattern built from `alias`: an alias can
    contain `%` or `_`, which are `LIKE` wildcards, so a built pattern would silently match
    the wrong thing instead of raising an error.

    When more than one alias appears in `text`, returns the SINGLE LONGEST match — a longer
    alias is more deliberate, less likely to be a coincidence (see `MIN_ALIAS_MATCH_LENGTH`).
    A question naming two entities at once is not what this function is for; the caller falls
    back to ordinary top-k retrieval."""
    query = select(GoldAlias.entity_type, GoldAlias.entity_id, GoldAlias.alias).where(GoldAlias.tenant == tenant)
    if entity_type is not None:
        query = query.where(GoldAlias.entity_type == entity_type)
    rows = (await session.execute(query)).all()

    lowered_text = text.lower()
    best: tuple[str, str, str] | None = None  # (entity_type, entity_id, alias)
    for row_entity_type, row_entity_id, alias in rows:
        if len(alias) < MIN_ALIAS_MATCH_LENGTH or alias.lower() not in lowered_text:
            continue
        if best is None or len(alias) > len(best[2]):
            best = (row_entity_type, row_entity_id, alias)
    if best is None:
        return None

    matched_entity_type, _, matched_alias = best
    resolved_id = await resolve_entity_id_for_lookup(session, matched_entity_type, matched_alias, tenant=tenant)
    if resolved_id is None:
        return None  # Defensive only: the alias row we just read should always resolve.
    return matched_entity_type, resolved_id, matched_alias


def _evolution_step_line(row: GoldEvolution) -> str:
    author = row.authored_by or "an unknown author"
    # `valid_to` lets an evolution answer say how long a version lasted, not just when it
    # started.
    validity = "still in effect" if row.valid_to is None else f"in effect until {row.valid_to.isoformat()}"
    return (
        f"- [{row.entity_type}:{row.entity_id}] Version {row.version}, {row.ingestion_date.isoformat()}, "
        f"by {author}, from ADR {row.source_component} v{row.source_adr_version} "
        f"(operation={row.operation}, {validity}): {row.narrative}"
    )


async def answer_evolution_question(
    question: str, canonical_name: str, rows: list[GoldEvolution], history: list[tuple[str, str]] | None = None
) -> GroundedAnswer:
    """Drafts a structured answer to a "how has X evolved" question, from EVERY version
    `entity_history` returned for one entity, oldest first — plain-text `answer` plus
    `citations`, checked against `rows` by `_verify_citations` the same way `answer_question`
    does. A separate function from `answer_question` on purpose: that one answers from a
    possibly incomplete top-k row set and prefers [latest version]; this one narrates the
    complete ordered sequence of changes.

    Each step's line names its entity, version, `ingestion_date`, author, the ADR it came
    from, operation, and narrative. The prompt asks for chronological prose: who introduced
    the entity and why, then each later step. `rows` empty means the entity was never found;
    this still degrades safely instead of sending an empty prompt.

    `history`, when given, is the most recent turns of this chat session — it is what let the
    caller resolve `canonical_name` for a follow-up like "how has it evolved?" that never
    names the entity. Passed here only so the answer matches the conversation's phrasing,
    never as a source of facts."""
    if not rows:
        return GroundedAnswer(answer=f"No recorded history was found for {canonical_name!r}.")
    timeline = "\n".join(_evolution_step_line(row) for row in rows)
    messages = [
        {
            "role": "user",
            "content": (
                "Write a chronological, narrative answer to the question below, using ONLY the "
                "timeline of versions given — do not invent anything the timeline doesn't "
                "state. Cover every step in order: who made each change, when, and why, "
                "reading naturally as a short history, not a bare list. If a step's author is "
                "\"an unknown author\", say the author is not recorded for that step instead of "
                "guessing one. Be thorough but not padded — 3-6 sentences for a short history, "
                "more only if the timeline genuinely has that many distinct steps worth "
                "naming. Each step also names the exact ADR it came from (source_component and "
                "version) — give that back verbatim if the question asks which ADR/document a "
                "step comes from, so the reader can look that exact ADR up in the Architecture "
                "history table. If a previous conversation is given below, use it only to phrase the "
                "answer naturally as part of that conversation — never as a source of facts on "
                "its own; every fact must still come from the timeline.\n\n"
                "Return a JSON object with two fields: \"answer\" (the text answer, as "
                "described above) and \"citations\" (a list of every step you actually used, "
                "each as {\"entity_type\": ..., \"entity_id\": ..., \"version\": ...} — copy "
                "these three values verbatim from the \"[entity_type:entity_id]\" tag and "
                "\"Version\" of each step you cite, never invent or guess one). A step you did "
                "not end up using in the answer does not belong in citations.\n\n"
                f"{_format_history_block(history)}"
                f"Entity: {canonical_name}\n\nTimeline (oldest first):\n{timeline}\n\n"
                f"Question: {question}"
            ),
        }
    ]
    response = await router.complete(messages, response_format=GroundedAnswer, temperature=0, reasoning_effort="none")
    result = GroundedAnswer.model_validate(load_json_response(response.choices[0].message.content))
    return GroundedAnswer(answer=result.answer, citations=_verify_citations(result.citations, rows))


__all__ = [
    "ArchitecturePayload",
    "ComponentPayload",
    "DEFAULT_MAX_DISTANCE",
    "DataContractPayload",
    "FUZZY_MATCH_THRESHOLD",
    "MAX_HISTORY_MESSAGES",
    "MIN_ALIAS_MATCH_LENGTH",
    "RECALL_POOL_SIZE",
    "RRF_K_CONSTANT",
    "already_extracted",
    "answer_evolution_question",
    "answer_question",
    "apply_contract_ripple_effect",
    "build_context_lines",
    "build_relationship_diagram",
    "classify_contract_directions",
    "content_hash",
    "contracts_with_real_changes",
    "current_architecture_diagram",
    "current_architecture_diagram_interactive",
    "current_gold_state",
    "current_gold_state_as_of",
    "embed_question",
    "ensure_alias",
    "entity_history",
    "expand_question",
    "extract_and_persist_gold_facts",
    "extract_gold_facts_for_source",
    "find_entity_by_name_in_text",
    "get_component_contracts",
    "get_predecessors",
    "get_successors",
    "gold_entities_for_adr",
    "is_evolution_question",
    "latest_odcs_spec",
    "latest_versions",
    "memo_key",
    "parse_odcs_spec",
    "persist_entity_version",
    "resolve_and_alias",
    "resolve_entity_id",
    "resolve_entity_id_for_lookup",
    "retrieve_with_correction",
    "top_k_gold_evolution",
    "wants_full_spec",
]
