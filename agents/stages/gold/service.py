"""This file holds reusable Gold-stage logic. Code can call it directly, for example a future
backfill script or unit tests, without needing the full LangGraph state shape. This mirrors the
split this repo's other stages already use relative to `agents/graph.py`: the three Gold nodes
in `agents/graph.py` (`extract_gold_facts`, `resolve_gold_identity`, `persist_gold_evolution`)
are thin wrappers that call the functions here.

Gold's per-ADR reconciliation runs inside Silver's own graph run. It does not run as a
separately triggered pipeline that watches `silver_documents` for changes. See
`.tmp/gold_process_v5.md` §1-2 for why.

Every function below is still a plain function that takes explicit arguments. None of them take
a graph `state`. This lets `.tmp/gold_process_v5.md` §4's standalone rebuild path call the exact
same functions the graph does. That rebuild path is a future `scripts/backfill_gold.py`: it
replays Gold over existing `silver_documents` rows, without re-running Silver's ACB loop.
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from datetime import date
from typing import Literal

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from agents.shared import extract_authors_line, transcription_base_name
from agents.stages.gold.prompts import build_gold_extraction_prompt
from agents.stages.gold.schemas import (
    ArchitecturePayload,
    ComponentPayload,
    DataContractPayload,
    GoldEntityType,
    GoldExtractionResult,
    GoldOperation,
)
from agents.template import load_json_response
from db.models import GoldAlias, GoldEvolution
from ingestion.embedder import embed
from llm import router

# `resolve_entity_id` uses this same value on either side of an ambiguous match. A lower value
# would start resolving genuinely different names to the same entity_id. For example, on a bad
# day it could merge "Order Service" and "Orders API" into one entity. A higher value would mint
# duplicate entity_ids for trivial spelling or casing drift, which is exactly what `gold_aliases`
# exists to absorb. This value is not tuned against a real corpus yet. It is a candidate to
# revisit once real transcripts exercise this path.
FUZZY_MATCH_THRESHOLD = 0.6


def content_hash(content: str) -> str:
    """`_entity_hash` below and `agents.graph._persist_document_version` both use this function.
    It is the one definition of "how we turn a string into our version hash." Silver's
    whole-document hash and Gold's per-entity hash both use it this way, even though they hash
    different-shaped things (v6 §4)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_odcs_spec(raw: str) -> dict:
    """Parses `ExtractedDataContract.odcs_spec`'s JSON-encoded string back into the dict that
    `DataContractPayload.odcs_spec` actually stores. See `ExtractedDataContract.odcs_spec`'s own
    docstring for why it is a string at all on the extraction side. In short: OpenAI's strict
    structured-output mode cannot express a deliberately open object field.

    Both real callers of this conversion share this one function: `extract_and_persist_gold_facts`
    below, and `agents.graph._persist_contracts`. Sharing it this way keeps the parsing and
    fallback rule from drifting apart between the two callers.

    This function falls back to `{}` for anything that is not a JSON object. A malformed spec
    string should degrade to "no spec captured" for that one contract. It should never crash Gold
    persistence for the whole ADR just because one contract has a formatting slip."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _entity_hash(operation: GoldOperation, narrative: str, payload: dict) -> str:
    """This uses the same hash-compare-then-bump mechanism as `content_hash` and
    `agents.graph._persist_document_version`. It is scoped to one entity's own fields instead of
    a whole document. See `GoldEvolution.entity_hash`'s column comment for why this function is
    not named `content_hash`.

    `payload` is serialized with `sort_keys=True`. This makes two logically identical payloads
    hash the same, even if their keys were inserted in a different order. This is a real risk
    here, because `payload` round-trips through Pydantic's `.model_dump()` in some call sites and
    through plain dict construction in others.

    `sort_keys=True` only normalizes dict key order. It does NOT normalize list element order. So
    `payload["dependency_ids"]`, `payload["contract_ids"]`, and similar list fields must already
    be sorted by the caller before this function runs. See `agents.graph.persist_gold_evolution`
    and `scripts.backfill_gold._backfill_one` for where that sorting happens. Without it, two
    logically identical extractions could list the same names in a different order. That would
    hash differently and trigger a version bump that should not happen."""
    canonical_payload = json.dumps(payload, sort_keys=True)
    return content_hash(f"{operation}\n{narrative}\n{canonical_payload}")


async def extract_gold_facts_for_source(adr_content: str) -> GoldExtractionResult:
    """This is one structured-extraction LLM call, using `temperature=0`. This is a deterministic
    extraction task, not creative writing. This follows the same reasoning as
    `agents.stages.architecture_questions.service.generate_architecture_questions_for_batch`'s
    own `temperature=0, reasoning_effort="none"` call.

    `reasoning_effort="none"` is required alongside `temperature=0`, for the same documented
    reason. A reasoning-locked model, for example gpt-5.6-terra, otherwise rejects any
    temperature value other than 1. Setting `reasoning_effort="none"` turns that check off.

    Anthropic's reasoning-locked models, for example claude-opus-5, have no equivalent setting to
    turn that check off. They hard-require temperature=1. If `settings.llm_fallback_order` ever
    falls through to Anthropic for this call, the call fails instead of falling back cleanly.
    This is the same known risk as the call this mirrors, and it is accepted for now, not fixed
    here. Fixing it would mean either reordering the fallback for every deterministic call in this
    codebase, or teaching the router itself to drop `temperature` and `reasoning_effort` per
    provider. Both are a bigger change than this pass's scope.

    This function takes only `adr_content`. It does not take a pre-given list of mentioned
    components or contracts. Gold discovers every component and contract straight from the
    final, clarified ADR. See `build_gold_extraction_prompt`'s own docstring for why grounding
    extraction against an earlier, pre-clarification list used to silently drop anything
    introduced only through a clarification answer."""
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


async def _lookup_entity_id(
    session: AsyncSession, entity_type: GoldEntityType, name: str, *, tenant: str
) -> str | None:
    """The shared exact-then-fuzzy lookup both `resolve_entity_id` and
    `resolve_entity_id_for_lookup` build on. First, an exact match on `gold_aliases.alias`.
    Second, a `pg_trgm` fuzzy match above `FUZZY_MATCH_THRESHOLD`, where the best similarity
    wins. Returns `None` if neither matches — what happens next (mint a new id, or report "no
    such entity") is the caller's decision, not this function's, since the write path and the
    read path need opposite answers to "nothing matched."

    `tenant` scopes both lookups. This is the one place where a missing tenant filter would cause
    a real data leak, not just a wrong count. Without this filter, tenant A's "Order Service"
    alias would exact-match or fuzzy-match tenant B's own "Order Service" mention. That would
    silently merge two unrelated companies' components under the same entity_id."""
    exact = (
        await session.execute(
            select(GoldAlias.entity_id).where(
                GoldAlias.tenant == tenant, GoldAlias.entity_type == entity_type, GoldAlias.alias == name
            )
        )
    ).scalars().first()
    if exact is not None:
        return exact

    fuzzy = (
        await session.execute(
            select(GoldAlias.entity_id)
            .where(
                GoldAlias.tenant == tenant,
                GoldAlias.entity_type == entity_type,
                func.similarity(GoldAlias.alias, name) > FUZZY_MATCH_THRESHOLD,
            )
            .order_by(func.similarity(GoldAlias.alias, name).desc())
            .limit(1)
        )
    ).scalars().first()
    return fuzzy


async def resolve_entity_id(
    session: AsyncSession, entity_type: GoldEntityType, name: str, *, tenant: str = "default"
) -> str:
    """This is the write-path identity resolution `resolve_and_alias` uses while persisting a
    fresh extraction. It tries `_lookup_entity_id` first. If neither an exact nor a fuzzy match
    exists, it mints a new `entity_id` as a ULID string (`python-ulid`), since a genuinely new
    entity must get an id regardless. `gold_evolution` and `gold_aliases` never treat this id as
    anything but an opaque string. This follows the same "owns nothing, purely a lookup"
    principle that `gold_process.md` §3 states for `gold_aliases`.

    A ULID, not a `uuid4`, is what this mints: 26 characters, Crockford-base32 (alphanumeric,
    case-insensitive, no ambiguous characters), lexicographically sortable by creation time. It
    is still just an opaque identity here, the same as a `uuid4` string would be — nothing reads
    a ULID's embedded timestamp or relies on its sort order. The reason to use one anyway is
    that this id is meant to appear in a component's own metadata and in a data contract's
    `producer_id`/`consumer_id` (see `DataContractPayload`), where a short, readable,
    unambiguous token is worth more than a `uuid4`'s hyphens and mixed-looking hex.

    This function does not insert the alias row itself. See `ensure_alias` for that. A name that
    resolves to an EXISTING entity_id through fuzzy match still needs its own alias row inserted,
    since it is a new variant of a known entity. That insert is a different operation from
    minting a new entity_id.

    Never reuse this function for a read-only lookup (a chat question, a report). Minting a
    fresh, alias-less id for "no match found" is only correct for a caller about to persist a
    version under it. A read-only caller wants `resolve_entity_id_for_lookup` instead, which
    reports "nothing matched" honestly instead of returning an empty new identity."""
    existing = await _lookup_entity_id(session, entity_type, name, tenant=tenant)
    return existing if existing is not None else str(ULID())


async def resolve_entity_id_for_lookup(
    session: AsyncSession, entity_type: GoldEntityType, name: str, *, tenant: str = "default"
) -> str | None:
    """The read-only counterpart to `resolve_entity_id`, for a caller that is asking "does this
    entity already exist," not "give me an id to write under." Returns `None` when neither an
    exact nor a fuzzy alias match exists, instead of minting a fresh ULID the way the write
    path does — a fresh id would have zero `gold_evolution` rows on record, which would read as
    "found it, but it has no history" instead of the true "no such entity was ever seen."
    `find_entity_by_name_in_text` uses this to resolve the component or contract name it found
    in a question, before fetching that entity's full evolution history."""
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
    """Inserts this `(tenant, entity_type, entity_id, alias)` row if it does not already exist.
    This uses `ON CONFLICT DO NOTHING`. This is the same idempotent-insert convention the rest of
    this pipeline uses for anything that can legitimately re-run over the same source, for
    example `agents.graph._persist_document_version`'s overwrite branch.

    This function is called both for a brand-new entity_id's first alias, and for an existing
    entity_id seen under a new name variant. Both cases just need an insert-if-absent. Neither
    case needs different logic."""
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
    """This composes `resolve_entity_id` and `ensure_alias`. It is memoized against the caller's
    own source-scoped `name_to_id` map. So a name already resolved earlier in the same batch is
    just a dict lookup, not a second round trip to the database.

    Both `agents.graph.resolve_gold_identity` and `scripts.backfill_gold._backfill_one` need this
    same identity-resolution step. Before this function existed, it was a nested closure in the
    first one and a near-identical module-level function in the second one. Now both just call
    this single definition."""
    if name in name_to_id:
        return name_to_id[name]
    entity_id = await resolve_entity_id(session, entity_type, name, tenant=tenant)
    name_to_id[name] = entity_id
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
) -> int | None:
    """This does a hash-compare-then-bump against the latest `gold_evolution` row for this
    `(entity_type, entity_id)`. It runs a query (`ORDER BY version DESC LIMIT 1`), not a
    materialized cache lookup, because `gold_current_state` is cut for this pass
    (`.tmp/optmizaciones.md` §1).

    This uses the same mechanism as `agents.graph._persist_document_version`, scoped to one
    entity. If the `entity_hash` is identical to the latest row, this is a no-op and returns
    `None`. If it is different, or there is no prior row at all, this inserts `version + 1` (or
    `version = 1` for the first row). It embeds `narrative` using the same `ingestion/embedder.py`
    singleton that Bronze and Silver already use. There is no second embedding stack.

    `authored_by` is not part of `_entity_hash`, the same way `source_component`,
    `source_adr_version`, and `ingestion_date` already are not: it describes where this version
    came from, not what changed. Two extractions with the same operation, narrative, and payload
    still hash identically and no-op, even if `authored_by` somehow differed between them.

    It is the caller's responsibility, not this function's, to never call this for an entity
    where `operation == "unknown"`. See `agents.graph.persist_gold_evolution`'s skip step, which
    runs before reaching this function. `GoldOperation` types `operation` as the union of all
    three per-`entity_type` literals. This union still includes `"unknown"` at the type level,
    because Python cannot easily express "minus one value." This function does not itself check
    that `operation` is the right *subset* for the given `entity_type`. That pairing is the
    caller's responsibility."""
    entity_hash = _entity_hash(operation, narrative, payload)
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

    if latest is not None and latest.entity_hash == entity_hash:
        return None

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
    """Extracts Gold facts from one finished ADR, and saves every resolved component, contract,
    and architecture entity. This is the shared body two callers outside the LangGraph flow need:
    `scripts/backfill_gold.py`, a standalone rebuild script, and the frontend's "Publish" finalize
    endpoint (`app/routers/frontend.py`). That endpoint saves a reviewer-approved, possibly
    regenerated ADR that never went through the graph's own `write_document` or Gold nodes at all.

    This function does NOT replace the three graph nodes (`extract_gold_facts`,
    `resolve_gold_identity`, `persist_gold_evolution` in `agents/graph.py`). Those three stay as
    separately traced Logfire spans on purpose, matching every other node in that graph. This
    function exists for the two callers that were never part of that traced graph run to begin
    with. Before this function existed, those two callers duplicated this same
    persist-orchestration logic between themselves. See `scripts/backfill_gold.py`'s own former
    "KNOWN DUPLICATION" note.

    This function skips entirely, returning `False` with no LLM call, if this exact `(tenant,
    source_component, source_adr_version)` was already extracted. This gives the same idempotency
    that `agents.graph.extract_gold_facts` already gives a re-run of the same content.

    `force=True` re-runs the LLM call regardless. Use this, for example, after changing the
    extraction prompt or model and wanting to reprocess history. Even with `force=True`,
    `persist_entity_version`'s own hash-compare-then-bump still no-ops any entity whose result
    comes back identical. This function returns `True` when it actually extracted."""
    if not force and await already_extracted(session, source_component, source_adr_version, tenant=tenant):
        return False

    result = await extract_gold_facts_for_source(adr_content)
    # `adr_content` already carries its own `**Authors:** <username>` line by the time any of
    # this function's three real callers reach here — `insert_authors_line` always runs first,
    # in `agents.graph.write_document`, `app.routers.frontend.finalize_document`, and whatever
    # wrote the `silver_documents.content` row `scripts/backfill_gold.py` replays. So reading it
    # back here, the same way `agents.graph._persist_document_version` already does for
    # `silver_documents.authored_by`, needs no extra parameter and no second source of truth.
    authored_by = extract_authors_line(adr_content)
    name_to_id: dict[str, str] = {}

    for component in result.components:
        if component.status == "unknown":
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
        if contract.action == "unknown":
            continue
        await resolve_and_alias(
            session, "data_contract", contract.name, name_to_id, source_component, source_adr_version, tenant=tenant
        )
        for component_name in (contract.producer, contract.consumer):
            await resolve_and_alias(
                session, "component", component_name, name_to_id, source_component, source_adr_version, tenant=tenant
            )

    for component in result.components:
        if component.status == "unknown":
            continue
        payload = ComponentPayload(
            dependency_ids=sorted({name_to_id[n] for n in component.dependency_names if n in name_to_id}),
            contract_ids=sorted({name_to_id[n] for n in component.contract_names if n in name_to_id}),
        ).model_dump()
        await persist_entity_version(
            session,
            entity_type="component",
            entity_id=name_to_id[component.name],
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
        if contract.action == "unknown":
            continue
        payload = DataContractPayload(
            producer=contract.producer,
            consumer=contract.consumer,
            producer_id=name_to_id.get(contract.producer, ""),
            consumer_id=name_to_id.get(contract.consumer, ""),
            odcs_spec=parse_odcs_spec(contract.odcs_spec),
        ).model_dump()
        await persist_entity_version(
            session,
            entity_type="data_contract",
            entity_id=name_to_id[contract.name],
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

    architecture_payload = ArchitecturePayload(
        mermaid_diagram=result.mermaid_diagram,
        components=sorted({c.name for c in result.components}),
        dependencies=sorted({dep for c in result.components for dep in c.dependency_names}),
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
    """This is the "current state" read path that `v4`'s now-cut `gold_current_state` table would
    have served. It returns one row per `(entity_type, entity_id)`, always the latest version,
    as a query over `gold_evolution` instead of a maintained cache (`.tmp/optmizaciones.md` §1).

    Postgres `DISTINCT ON` needs its own `ORDER BY` prefix that matches the `DISTINCT ON` columns,
    before the tie-breaking `version DESC`. So this uses raw column ordering. The ORM's
    `.distinct()` does not express this directly.

    This always scopes to one `tenant`. This is the read path for the Chat tab and for
    `current_architecture_diagram`. Neither of those should ever see another tenant's
    architecture."""
    query = select(GoldEvolution).distinct(GoldEvolution.entity_type, GoldEvolution.entity_id).where(
        GoldEvolution.tenant == tenant
    )
    if entity_type is not None:
        query = query.where(GoldEvolution.entity_type == entity_type)
    query = query.order_by(GoldEvolution.entity_type, GoldEvolution.entity_id, GoldEvolution.version.desc())
    return (await session.execute(query)).scalars().all()


async def current_architecture_diagram(session: AsyncSession, *, tenant: str = "default") -> str:
    """Builds a Mermaid `flowchart LR` from Gold's own current component state. This is
    deliberately NOT the LLM-drawn diagram that any single ADR's `entity_type="architecture"` row
    carries (`ArchitecturePayload.mermaid_diagram`). That diagram only ever reflects what ONE ADR
    asserted. This function reads `current_gold_state`'s live view instead. So it reflects every
    component Gold currently has on record across every ADR, with any `removed` component
    dropped.

    This backs the frontend's "Architecture history" tab. Each node's label carries a small
    italic subtitle naming the `(source_component, source_adr_version)` that last touched it.
    This returns `""` if Gold has no live component yet.

    Navigating to a node's ADR is handled by plain "View ADR" links in the table that
    `frontend/app.py`'s `_architecture_history_tab` renders below the diagram. This is
    deliberately NOT a `click <id> href ...` directive on the node itself. See below for why.

    Two node-label choices are a defensive response to a real, reported bug. `st.mermaid_chart`
    used to render every node as an invisible, zero-size box, with only the connecting edges
    visible:

    1. The subtitle uses Mermaid's own backtick "markdown string" label syntax: a literal newline
       plus `*italic*`. This is the officially supported way to get a multi-line styled label,
       instead of raw `<br/>` or `<sub>` HTML, which depends on how a given renderer treats
       arbitrary embedded HTML.
    2. Every node gets an explicit `classDef` and `class` for fill, stroke, and text color,
       instead of relying on Mermaid's own default theme fill. This is the same reasoning that
       `prompts/adr_generation/generator.jinja`'s DIAGRAM COLOR CODING section already applies to
       ADR diagrams: never trust ambient or default coloring to stay visible across renderers.

    Neither of those two choices was the actual cause of the bug. Here is the ROOT CAUSE,
    confirmed by driving a real headless Chrome against the live Streamlit app through
    `st.mermaid_chart`, then fetching and inspecting the `blob:` SVG it renders to: a
    `click <id> href "..." "_blank"` directive was present in every earlier version of this
    diagram, to make a node open its ADR. That directive makes `st.mermaid_chart` render
    `<g class="nodes"/>` completely empty, with zero node elements, while it still renders the
    edge paths. This exactly matches the "only edges visible" symptom.

    This was reproduced in isolation: a single-word plain-text node with a `click ... href` line,
    no classDef, and no markdown label fails the same way. The same diagram with the `click` line
    removed renders every node correctly. Standalone `mermaid-cli`, run under an equivalent
    `securityLevel: "strict"` config, does NOT reproduce this bug. So `st.mermaid_chart`'s own
    render pipeline is where interactivity directives specifically break. The most likely reason:
    Mermaid's click-binding step needs the SVG attached to the live document to resolve node ids,
    but Streamlit's component renders off-screen before converting the result to a blob image.

    Do not add `click` directives back into this diagram without re-verifying against a real
    running `st.mermaid_chart`, not just `mermaid-cli`. That gap is exactly what let this bug
    ship broken multiple times before.

    Edges are drawn as `component --> dependency`, from `ComponentPayload.dependency_ids`. A
    dependency that is not itself a live component, because it was removed or never resolved, is
    silently skipped. It is not drawn as a dangling node."""
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
        adr_ref = f"{transcription_base_name(c.source_component)} v{c.source_adr_version}"
        lines.append(f'    {node_id[c.entity_id]}["`{c.canonical_name}\n*{adr_ref}*`"]')
    for c in live:
        dependency_ids = ComponentPayload.model_validate(c.payload).dependency_ids
        for dependency_id in dependency_ids:
            if dependency_id in node_id and dependency_id != c.entity_id:
                lines.append(f"    {node_id[c.entity_id]} --> {node_id[dependency_id]}")
    lines.append(f"    class {','.join(node_id.values())} goldNode")
    return "\n".join(lines)


# --- Top-k retrieval. This is the RAG-consumer read path that `current_gold_state` above does
# not cover. `current_gold_state` answers "what is the latest version per entity." This answers
# "which versions, across the whole history, are semantically closest to a question." Both
# `scripts/chat_gold.py` (interactive use) and `agents/stages/gold/testing/retrieval.py` (the
# suite that asserts against this) share this same retrieval mechanism, so it has exactly one
# definition. -----------------------------------------------------------------------------------

# This is the cosine distance (1 - cosine_similarity) above which a row gets dropped from
# top-k results, instead of being forced into the answer. Plain top-k always returns exactly
# `k` rows, even when none of them are actually relevant to the question. That is how a RAG
# answer ends up confidently invented from the least-bad match, instead of saying "I don't
# know." This has the same caveat as `FUZZY_MATCH_THRESHOLD` above: it is not tuned against a
# real corpus yet. It is a starting point to revisit once real chat usage gives us distances to
# calibrate against.
#
# `max_distance` is opt-in on `top_k_gold_evolution`, defaulting to `None` (unfiltered). This is
# on purpose, so this default value can change later without silently changing what
# `agents.stages.gold.testing`'s already real-LLM-verified assertions see.
DEFAULT_MAX_DISTANCE = 0.6

# `mode="hybrid"` fetches this many candidates from EACH ranking (vector, lexical) before fusing
# them, not just `k`. RRF needs depth on both sides to do anything useful: a row that ranks,
# say, 7th by embedding similarity and 2nd by lexical match should still be able to win a fused
# top-5 over two rows that only rank in the top 5 of one signal and nowhere in the other. Fusing
# two lists already cut to `k` would never give a moderate-vector/strong-lexical row that
# chance. This is deliberately NOT the same thing as the "expand-then-dedupe-by-entity" idea in
# `.tmp/advanced_techniques.md` — no deduplication happens here, only wider recall for fusion.
HYBRID_RECALL_POOL = 30

# Reciprocal Rank Fusion's own smoothing constant. 60 is the value the original RRF paper (Cormack
# et al., 2009) tuned against, and it is what most hybrid-search implementations still default
# to — a larger value flattens the gap between rank 1 and rank 30, a smaller one makes rank 1
# dominate almost completely. Nothing about this corpus has been measured against a different
# value yet, so this stays at the well-established default rather than an invented number.
RRF_K_CONSTANT = 60


async def embed_question(text: str) -> list[float]:
    """`ingestion.embedder.embed` is a blocking, batched call. This wraps it in
    `asyncio.to_thread`, the same way `persist_entity_version` already does, applied here to one
    question at a time."""
    [vector] = await asyncio.to_thread(embed, [text])
    return vector


def _reciprocal_rank_fusion(rankings: list[list[int]], k_constant: int = RRF_K_CONSTANT) -> list[int]:
    """Fuses any number of ranked id lists into one combined ranking. Each id's fused score is
    the sum, across every input ranking it appears in, of `1 / (k_constant + rank)` (rank
    counted from 1). An id absent from one of the rankings contributes nothing from it — not a
    penalty. That is the entire point of fusing instead of intersecting: a row only needs to
    rank well in ONE of the two signals (semantic OR lexical) to surface near the top, since a
    row that is the single best lexical match but a mediocre embedding match (an exact proper
    noun, an acronym) is exactly the case hybrid search exists to rescue.

    Returns ids sorted by fused score, descending. Ties (an id with the same fused score as
    another, which only realistically happens for two ids appearing in neither ranking) keep
    Python's stable sort order — the order they were first seen in `rankings`."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k_constant + rank)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


async def _top_k_ids_by_vector(
    session: AsyncSession,
    vector: list[float],
    limit: int,
    source_component: str | None,
    max_distance: float | None,
    tenant: str,
) -> list[int]:
    distance = GoldEvolution.embedding.cosine_distance(vector)
    query = select(GoldEvolution.id).where(GoldEvolution.tenant == tenant).order_by(distance).limit(limit)
    if source_component is not None:
        query = query.where(GoldEvolution.source_component == source_component)
    if max_distance is not None:
        query = query.where(distance <= max_distance)
    return list((await session.execute(query)).scalars().all())


async def _top_k_ids_by_lexical_rank(
    session: AsyncSession,
    question_text: str,
    limit: int,
    source_component: str | None,
    tenant: str,
) -> list[int]:
    """The lexical half of hybrid search: Postgres full-text search over `search_vector` (see
    that column's own comment in `db/models.py` and migration `188c1b98dd96`), ranked by
    `ts_rank`. `plainto_tsquery` treats `question_text` as plain text, not `tsquery` syntax — a
    question typed by a person is not a search-operator expression, and letting `&`/`|`/`!`
    characters in a question be interpreted as tsquery operators would be a second, unrelated
    injection surface. A question that reduces to an empty tsquery (all stopwords, or no
    recognized lexemes) matches nothing here — RRF then falls back to whatever the vector
    ranking alone found, which is the correct degrade: no lexical signal is not an error."""
    tsquery = func.plainto_tsquery("english", question_text)
    query = (
        select(GoldEvolution.id)
        .where(GoldEvolution.tenant == tenant, GoldEvolution.search_vector.op("@@")(tsquery))
        .order_by(func.ts_rank(GoldEvolution.search_vector, tsquery).desc())
        .limit(limit)
    )
    if source_component is not None:
        query = query.where(GoldEvolution.source_component == source_component)
    return list((await session.execute(query)).scalars().all())


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
) -> list[GoldEvolution]:
    """This runs top-k search over ALL versions of `gold_evolution`, not just the latest version
    per entity. Getting the latest version per entity is `current_gold_state`'s job instead.

    This always scopes to one `tenant` first. The Chat tab must never retrieve, let alone answer
    from, another tenant's architecture facts. `source_component`, when given, narrows the search
    to one source within that tenant. When omitted, it searches the whole tenant's rows.

    `max_distance`, when given (see `DEFAULT_MAX_DISTANCE`), drops rows past that distance instead
    of always returning `k` rows regardless of relevance. This filter runs in SQL, not after the
    fact, so a tight bound also means less data fetched over the wire. It only ever applies to
    the vector-distance ranking — `ts_rank`'s scale is not comparable to cosine distance, so there
    is no equivalent bound on the lexical side; a question with no lexical match simply
    contributes zero lexical candidates, handled by `_reciprocal_rank_fusion` itself.

    `mode="vector"` (the default, unchanged from before hybrid search existed) orders purely by
    `.cosine_distance(vector)`, matching `GoldEvolution.embedding`'s HNSW index
    (`vector_cosine_ops`) so it can actually use it. `mode="hybrid"` additionally runs a lexical
    `ts_rank` search over `search_vector` and fuses both rankings with Reciprocal Rank Fusion
    (`_reciprocal_rank_fusion`) before cutting to `k` — see `.tmp/advanced_techniques.md` §1 for
    why: an embedding can blur an exact component name, acronym, or ODCS field name that a plain
    keyword match finds immediately, at close to zero extra cost (the GIN index, versus the
    embedding API/model call already being paid for `vector`). `mode="hybrid"` requires
    `question_text` — the original question, not `vector`'s embedding of it, since the lexical
    branch does its own tokenization, never the embedding."""
    if mode == "vector":
        ids = await _top_k_ids_by_vector(session, vector, k, source_component, max_distance, tenant)
    elif mode == "hybrid":
        if not question_text:
            raise ValueError("mode='hybrid' requires question_text (the lexical ranking needs the raw question)")
        # Sequential, not `asyncio.gather` — a single `AsyncSession` cannot run two queries
        # concurrently over one DBAPI connection. Doing so raised a real
        # `sqlalchemy.exc.IllegalStateChangeError` here, confirmed against the actual running
        # app, not just a theoretical concern: two coroutines sharing `session` inside
        # `gather` both took the connection into an "in progress" state at once, and closing
        # the session afterward hit that half-finished state.
        vector_ids = await _top_k_ids_by_vector(session, vector, HYBRID_RECALL_POOL, source_component, max_distance, tenant)
        lexical_ids = await _top_k_ids_by_lexical_rank(session, question_text, HYBRID_RECALL_POOL, source_component, tenant)
        ids = _reciprocal_rank_fusion([vector_ids, lexical_ids])[:k]
    else:
        raise ValueError(f"unknown top_k_gold_evolution mode: {mode!r}")

    if not ids:
        return []
    rows = (await session.execute(select(GoldEvolution).where(GoldEvolution.id.in_(ids)))).scalars().all()
    rows_by_id = {row.id: row for row in rows}
    return [rows_by_id[item_id] for item_id in ids if item_id in rows_by_id]


async def latest_versions(session: AsyncSession, rows: Sequence[GoldEvolution]) -> dict[tuple[str, str], int]:
    """Returns the current, latest version number for each distinct `(entity_type, entity_id)`
    appearing in `rows`. This runs one batched query, not one lookup per row.

    This lets `answer_question` tell the LLM which retrieved rows are still current, and which
    are superseded by a later version that may not have made the same top-k list. Top-k search
    runs across ALL versions, so an older version's narrative can rank closer to the question
    than its own entity's current version does. See `top_k_gold_evolution`'s own docstring for
    more on that."""
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


async def answer_question(
    question: str, rows: list[GoldEvolution], latest: dict[tuple[str, str], int] | None = None
) -> str:
    """Drafts a plain-text answer to `question` from the retrieved rows. This is the
    RAG-consumer step that top-k retrieval alone does not cover. It returns plain text, with no
    `response_format`. This follows the same convention that `agents.stages.adr_generation.testing`'s own Actor call
    uses for its final ADR document: this prompt writes an answer directly, not structured JSON.

    `latest` comes from `latest_versions` and is optional. It tags each row as `[latest version]`
    or `[superseded — latest is version N]`, in the context the LLM sees. This stops a question
    about current state from getting answered using a row that ranked close by embedding
    similarity but has since been superseded. Without `latest`, every row is presented the same
    way. That is what happens today when a caller does not pass it."""
    if not rows:
        return "No relevant Gold facts were found for this question."
    context = "\n\n".join(
        f"- [{row.entity_type}] {row.canonical_name} (version {row.version}, "
        f"operation={row.operation}){_version_tag(row, latest)}: {row.narrative}"
        for row in rows
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Answer the question below using ONLY the retrieved architecture facts as "
                "context — do not invent anything the facts don't state. Be concise (2-4 "
                "sentences). If the facts describe an entity's evolution across versions, "
                "state its current/latest status explicitly, preferring facts tagged "
                "[latest version] for that; facts tagged [superseded] describe history, not "
                "the current state, and should only be used to answer questions about how "
                "something evolved over time.\n\n"
                f"Retrieved facts:\n{context}\n\nQuestion: {question}"
            ),
        }
    ]
    response = await router.complete(messages, temperature=0, reasoning_effort="none")
    return response.choices[0].message.content.strip()


# --- Entity evolution timeline. This is a third read path, next to `current_gold_state`
# ("what is true now") and `top_k_gold_evolution` ("which versions, across the whole history,
# are semantically closest to a question"). A question like "how has X evolved over time" is
# neither of those: it wants EVERY version of ONE already-identified entity, in order, not a
# similarity-ranked, possibly incomplete top-k slice. Top-k can legitimately drop an early
# version whose narrative just does not word-match the question, which silently truncates the
# very timeline the question is asking for. This path fetches deterministically instead: find
# the one entity the question names, then take its full history, no ranking involved.
# -------------------------------------------------------------------------------------------

# `find_entity_by_name_in_text` ignores any alias shorter than this. A short alias (an
# acronym, a one-word service name fragment) is too likely to appear in a question by
# coincidence, unrelated to that entity, and would otherwise win by being "the alias that
# happens to be in this text." Longer, more specific aliases are more trustworthy as a match.
MIN_ALIAS_MATCH_LENGTH = 4

# `is_evolution_question` triggers on any of these appearing in the question, case-insensitive,
# English and Spanish. This is a deliberately narrow, deterministic check, not an LLM
# classification call — matching this codebase's general preference for a mechanical check
# over a probabilistic judge wherever one is reliable enough (see, for example,
# `agents.stages.classification`'s own DUPLICATE QUESTIONS handling). A caller only takes the
# full-history path when this ALSO holds alongside a confident `find_entity_by_name_in_text`
# match — an ordinary factual question that happens to name a component (e.g. "who is the
# producer of X") must still get plain top-k retrieval, not a whole narrated timeline.
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
    """Returns `True` if `question` reads as asking for an entity's full history, in either
    English or Spanish — see `_EVOLUTION_QUESTION_MARKERS`. Intended to gate the full-history
    path (`entity_history` + `answer_evolution_question`) alongside a confident
    `find_entity_by_name_in_text` match, never on its own: naming an entity is not enough,
    since most questions that name one are ordinary factual lookups, not history requests."""
    lowered = question.lower()
    return any(marker in lowered for marker in _EVOLUTION_QUESTION_MARKERS)


async def entity_history(
    session: AsyncSession, entity_type: GoldEntityType, entity_id: str, *, tenant: str = "default"
) -> list[GoldEvolution]:
    """Every `gold_evolution` row ever written for this one `(entity_type, entity_id)`, oldest
    first. This is the deterministic counterpart to `top_k_gold_evolution`: no ranking, no
    cutoff, no risk of an early version falling outside some `k`. `answer_evolution_question`
    is the intended consumer — it needs the complete timeline, not a sample of it."""
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

    This checks containment in Python, not a SQL `LIKE`/`ILIKE` pattern built from `alias`. An
    alias is arbitrary user-sourced text — it can contain `%` or `_`, which are `LIKE`
    wildcards. Building a pattern from it would silently change what the pattern matches
    instead of raising an error, exactly the kind of bug that stays invisible until a real
    component happens to be named "A/B_Test". `gold_aliases` is a small lookup table per
    tenant (`GoldAlias`'s own docstring), so fetching every alias and checking containment in
    Python costs nothing extra worth avoiding a wildcard-injection risk for.

    When more than one alias appears in `text` (for example, both "Order Service" and its
    substring "Order" are aliases, or the question genuinely names two different components),
    this returns the SINGLE LONGEST match. A longer alias is a more specific, more deliberate
    match than a short one that could just as easily be a coincidence — see
    `MIN_ALIAS_MATCH_LENGTH`. A question that genuinely asks about two entities at once is not
    what this function is for; the caller falls back to ordinary top-k retrieval in that case.
    This resolves the match through `resolve_entity_id_for_lookup`, not the alias's own stored
    `entity_id` column directly, so a caller never receives an entity_id for a match this
    tenant's own alias table would not otherwise resolve to (defensive, since today those are
    the same value either way)."""
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
    return (
        f"- Version {row.version}, {row.ingestion_date.isoformat()}, by {author} "
        f"(operation={row.operation}): {row.narrative}"
    )


async def answer_evolution_question(question: str, canonical_name: str, rows: list[GoldEvolution]) -> str:
    """Drafts a plain-text answer to a "how has X evolved over time" question, from EVERY
    version `entity_history` returned for one entity, oldest first. This is deliberately a
    separate function from `answer_question`, not a shared one with an extra flag: that
    function answers from a top-k, possibly cross-entity, possibly incomplete row set, and
    tells the LLM to prefer the [latest version] tag for current-state questions. This
    function's whole point is the opposite — narrate the complete, ordered sequence of
    changes, not settle on one current answer.

    Each step's line names its version, event-time date (`ingestion_date`, when the change was
    asserted, not `changed_at`), author (`authored_by`, empty for a CLI/script/test run with no
    real user — see `GoldEvolution.authored_by`'s own column comment), operation, and narrative.
    The prompt asks for a chronological prose account: who introduced the entity, when, and
    why, then each later step and its own reason, in the style of "X introduced this on Y for
    Z, then evolved on V because...". `rows` empty means the entity was never found; the caller
    is expected to have already checked that via `find_entity_by_name_in_text` before calling
    this, but this function still degrades safely instead of sending an empty prompt."""
    if not rows:
        return f"No recorded history was found for {canonical_name!r}."
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
                "naming.\n\n"
                f"Entity: {canonical_name}\n\nTimeline (oldest first):\n{timeline}\n\n"
                f"Question: {question}"
            ),
        }
    ]
    response = await router.complete(messages, temperature=0, reasoning_effort="none")
    return response.choices[0].message.content.strip()


__all__ = [
    "ArchitecturePayload",
    "ComponentPayload",
    "DEFAULT_MAX_DISTANCE",
    "DataContractPayload",
    "FUZZY_MATCH_THRESHOLD",
    "HYBRID_RECALL_POOL",
    "MIN_ALIAS_MATCH_LENGTH",
    "RRF_K_CONSTANT",
    "already_extracted",
    "answer_evolution_question",
    "answer_question",
    "content_hash",
    "current_architecture_diagram",
    "current_gold_state",
    "embed_question",
    "ensure_alias",
    "entity_history",
    "extract_and_persist_gold_facts",
    "extract_gold_facts_for_source",
    "find_entity_by_name_in_text",
    "is_evolution_question",
    "latest_versions",
    "parse_odcs_spec",
    "persist_entity_version",
    "resolve_and_alias",
    "resolve_entity_id",
    "resolve_entity_id_for_lookup",
    "top_k_gold_evolution",
]
