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
from datetime import date, datetime
from typing import Literal

from sqlalchemy import Numeric, and_, func, or_, select, text, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from agents.shared import extract_authors_line, transcription_base_name
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
    ("output"), read straight off this same ADR's own contract extraction — no query needed.
    Shared by both real callers that build `ComponentPayload` (`extract_and_persist_gold_facts`
    below, `agents.graph._persist_components`), so the split logic cannot drift between them.
    `contracts` must be a list of plain dicts (one `ExtractedDataContract.model_dump()` per
    contract, or `extraction["contracts"]`'s own already-dict shape) — never `ExtractedDataContract`
    instances directly, since this indexes with `contract["field"]`."""
    directions: dict[str, dict[str, set[str]]] = {}
    for contract in contracts:
        if contract["action"] == "unknown" or contract["name"] not in name_to_id:
            continue
        contract_id = name_to_id[contract["name"]]
        consumer, producer = contract.get("consumer"), contract.get("producer")
        if consumer:
            directions.setdefault(consumer, {"input": set(), "output": set()})["input"].add(contract_id)
        if producer:
            directions.setdefault(producer, {"input": set(), "output": set()})["output"].add(contract_id)
    return directions


def contracts_with_real_changes(contracts: list, name_to_id: dict[str, str]) -> set[str]:
    """The entity ids of every contract in THIS SAME extraction whose own `action` is a real
    change — anything other than `"unchanged"`/`"unknown"`. Used to correct a component's own
    `"unchanged"` judgment to `"modified"` when a contract it was ALREADY associated with (same
    id, still in its `input_contract_ids`/`output_contract_ids` — its own list of ids did not
    change) itself got a new version underneath it this round. See `doc/cicle_evolution.md`'s
    "Regla especial", condition 2. Shares the same "plain dicts, not `ExtractedDataContract`
    instances" contract as `classify_contract_directions`, and both real callers
    (`extract_and_persist_gold_facts` below, `agents.graph._persist_components`) call this one
    alongside it, from the exact same `contracts` list."""
    return {
        name_to_id[c["name"]]
        for c in contracts
        if c["action"] not in ("unchanged", "unknown") and c["name"] in name_to_id
    }


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


async def gold_entities_for_adr(
    session: AsyncSession, source_component: str, source_adr_version: int, *, tenant: str = "default"
) -> Sequence[GoldEvolution]:
    """Returns every `gold_evolution` row this exact `(tenant, source_component,
    source_adr_version)` triple actually wrote — every Gold entity this specific ADR version
    created or changed. `persist_entity_version` is a hash-compare-then-bump no-op when an
    entity's asserted state is unchanged from its prior version, so an entity this ADR merely
    re-confirmed without altering has no row here; only entities this ADR version genuinely
    added or changed do. This is the same `(tenant, source_component, source_adr_version)`
    filter `already_extracted` uses to check existence — this returns the full rows instead, for
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
    """The shared exact-then-fuzzy lookup both `resolve_entity_id` and
    `resolve_entity_id_for_lookup` build on. First, an exact match on `gold_aliases.alias`.
    Second, a `pg_trgm` fuzzy match above `FUZZY_MATCH_THRESHOLD`, where the best similarity
    wins. Returns `None` if neither matches — what happens next (mint a new id, or report "no
    such entity") is the caller's decision, not this function's, since the write path and the
    read path need opposite answers to "nothing matched."

    `tenant` scopes both lookups. This is the one place where a missing tenant filter would cause
    a real data leak, not just a wrong count. Without this filter, tenant A's "Order Service"
    alias would exact-match or fuzzy-match tenant B's own "Order Service" mention. That would
    silently merge two unrelated companies' components under the same entity_id.

    The fuzzy comparison rounds `similarity()` to 4 decimal places before comparing it to
    `FUZZY_MATCH_THRESHOLD`, instead of comparing the raw value directly. `pg_trgm.similarity()`
    returns a 4-byte `real`, and `FUZZY_MATCH_THRESHOLD` is a Python `float` (8-byte double) —
    comparing `real > float8` widens the `real` to double precision first, and that widening can
    turn a conceptually exact 0.6 into something like `0.6000000238418579`, which then passes a
    strict `> 0.6` check it should not. This is not theoretical: `similarity('frontend',
    'frontend-intra')` is exactly this case, and without the rounding here, it silently merged
    two genuinely different components (a marketplace frontend and an unrelated, deliberately
    separate intranet frontend) into one Gold entity — confirmed by reproducing the exact query
    against the live database. Rounding first makes the comparison exact at the precision that
    actually matters (four decimal places is already far finer than this threshold needs to be
    tuned to), so a true 0.6 compares as 0.6, never as marginally more."""
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
    caller's responsibility.

    See `doc/cicle_evolution.md` for the full evolution rules this enforces. Two of them live
    here specifically:

    1. **The payload corrects a false "unchanged" for a component.** The LLM's `operation`
       judgment is about the entity's own NARRATED behavior, not its structured facts. A
       component's own responsibility text can stay put while its payload
       (`dependency_ids`/`input_contract_ids`/`output_contract_ids`) genuinely differs from the
       previous version — for example, it starts producing a brand new data contract. When that
       happens for `entity_type == "component"`, `operation` is corrected to `"modified"` before
       persisting, so the stored row never claims "unchanged" while the facts say otherwise. This
       correction does NOT apply to `data_contract`: `ContractAction` has no generic "modified" —
       it distinguishes `forward-update` from `break-change`, a semantic judgment about the
       schema change itself that cannot be inferred from a payload diff alone (see
       `agents.shared.ContractAction`'s own docstring on this being an unverified LLM call today).
    2. **A carried-forward narrative neutralizes pure re-wording noise.** Every ADR is a fresh
       LLM call, so "this is still unchanged" gets re-narrated in different words each time even
       when nothing real changed — confirmed as a real bug: several contracts in production data
       picked up 2-3 spurious versions in a row with byte-identical payloads, purely from wording
       drift. When `operation == "unchanged"` (after rule 1's correction, if any, has already
       run) AND a prior version exists, `narrative` itself — not just a value used for hashing —
       is REPLACED with the previous version's own narrative, and it is that replaced value that
       gets both hashed and (if a version still ends up being inserted, e.g. because the payload
       changed) stored. This must replace the actual stored value, not only what gets hashed: an
       earlier version of this fix hashed against the previous narrative but stored the fresh
       one, which is unsound — it silently made two different rows compare as identical to a
       THIRD row by two different, inconsistent narrative texts, breaking the very next
       comparison down the chain. Concretely, this means a component's displayed narrative, once
       "unchanged" starts repeating, stays pinned to the last version where something genuinely
       changed — never diluted by a string of "still nothing changed" rewrites — until a real
       change (payload, or a genuine `"modified"`/`"new"`) writes a fresh one again."""
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

    if operation == "unchanged" and latest is not None:
        # Replaces the actual value that gets hashed AND stored — not just a local variable used
        # for hashing — so the chain stays self-consistent for the NEXT comparison. See the
        # docstring above for why hashing against one narrative while storing another is unsound.
        narrative = latest.narrative

    entity_hash = _entity_hash(operation, narrative, payload)

    if latest is not None and latest.entity_hash == entity_hash:
        return None

    if latest is not None:
        # Closes the outgoing version's validity window the instant it is superseded — never
        # touched again afterward. See `GoldEvolution.valid_to`'s own comment in `db/models.py`.
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
    comes back identical. This function returns `True` when it actually extracted.

    `pg_advisory_xact_lock` below closes a real race in the `already_extracted` check above it:
    without a lock, two overlapping calls for the exact same `(tenant, source_component,
    source_adr_version)` — for example the frontend's "Publish" button double-clicked, sending
    two overlapping `POST /transcriptions/finalize` requests — can both read `already_extracted
    == False` before either has committed, so both go on to call the (non-deterministic) LLM
    extraction and both call `persist_entity_version` for every entity. `persist_entity_version`
    itself only no-ops on an EXACT hash match; two independent LLM calls rarely phrase a
    narrative byte-for-byte identically, so the second call's entities each look like a genuine,
    if spurious, new version — this was observed in practice as every entity from one ADR
    getting both a v1 and a v2. The lock serializes the whole check-extract-persist sequence per
    `(tenant, source_component, source_adr_version)`: a second overlapping call blocks here until
    the first commits, then re-reads `already_extracted` as `True` and returns immediately,
    instead of racing it. It is transaction-scoped (`_xact_`), so it releases itself at whatever
    commit or rollback ends this call's transaction — never held past this request. The frontend
    also disables the Publish button the instant it is clicked (see `frontend/app.py`'s
    `publish_key`), which prevents the common case from firing two requests at all; this lock is
    the guarantee for every other path into this function (a second browser tab, a retried
    request, a concurrent `scripts/backfill_gold.py` run), not a fallback for the frontend fix."""
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:lock_key), :version)"),
        {"lock_key": f"{tenant}:{source_component}", "version": source_adr_version},
    )
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

    contract_dicts = [c.model_dump() for c in result.contracts]
    contract_directions = classify_contract_directions(contract_dicts, name_to_id)
    changed_contract_ids = contracts_with_real_changes(contract_dicts, name_to_id)
    for component in result.components:
        if component.status == "unknown":
            continue
        directions = contract_directions.get(component.name, {"input": set(), "output": set()})
        # `doc/cicle_evolution.md` "Regla especial", condition 2 — see the identical comment in
        # `agents.graph._persist_components`, which this mirrors.
        status = component.status
        if status == "unchanged" and (directions["input"] | directions["output"]) & changed_contract_ids:
            status = "modified"
        payload = ComponentPayload(
            dependency_ids=sorted({name_to_id[n] for n in component.dependency_names if n in name_to_id}),
            contract_ids=sorted({name_to_id[n] for n in component.contract_names if n in name_to_id}),
            input_contract_ids=sorted(directions["input"]),
            output_contract_ids=sorted(directions["output"]),
        ).model_dump()
        await persist_entity_version(
            session,
            entity_type="component",
            entity_id=name_to_id[component.name],
            canonical_name=component.name,
            operation=status,
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
        odcs_spec = parse_odcs_spec(contract.odcs_spec)
        if not odcs_spec:
            odcs_spec = await latest_odcs_spec(session, name_to_id[contract.name], tenant)
        payload = DataContractPayload(
            producer=contract.producer,
            consumer=contract.consumer,
            producer_id=name_to_id.get(contract.producer, ""),
            consumer_id=name_to_id.get(contract.consumer, ""),
            odcs_spec=odcs_spec,
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


async def current_gold_state_as_of(
    session: AsyncSession, as_of: date, entity_type: GoldEntityType | None = None, *, tenant: str = "default"
) -> Sequence[GoldEvolution]:
    """Same read path as `current_gold_state`, pinned to whatever was valid on `as_of` instead of
    always the latest version. `valid_to` (closed by `persist_entity_version` the instant a newer
    version supersedes a row) makes each entity's version windows non-overlapping, so at most one
    row per `(entity_type, entity_id)` can match this filter — the trailing `version DESC` is a
    tie-breaker in principle only, never expected to matter in practice. See
    `.tmp/improve_timeline_questions_and_linage.md` §2.1."""
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
    same `(tenant, source_component, source_adr_version)` — not `row`'s own values directly.

    One ADR run persists several entities (components, contracts, the architecture snapshot) as
    separate, sequential inserts (`agents.graph.persist_gold_evolution`), so a component's own
    row can have a lower `id` than a contract from that EXACT SAME ADR that conceptually became
    true at the same moment. Cursoring on the component's own id would wrongly exclude that
    contract (and any other sibling from the same batch inserted after it) as "not yet known" —
    confirmed by a real test failure this fixed: a component with zero of its own dependencies,
    whose only relationship is a sibling from the same ADR that depends on it, showed no
    relationship at all when cursored on its own id. The batch's own last id is what correctly
    includes every fact from that same ADR, regardless of which one happened to be written
    first."""
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
    id)` cursor, ALWAYS compared as that one pair, never `ingestion_date` alone: several
    versions of the very same entity can legitimately share one `ingestion_date` (several ADRs
    uploaded the same day), which made `ingestion_date`-only comparisons unable to tell which of
    two same-day versions came first — a real, reported bug (a v1-pinned diagram still showed a
    dependency only added in v3, because both shared today's date). `id` alone would already be
    a perfect, gap-free insertion order; `changed_at` is included anyway because that is the
    pair this was explicitly asked to use, and comparing both together is exactly as correct as
    comparing `id` alone — `changed_at` only ever adds a tie-break that `id` had already settled.

    Used only when resolving a SPECIFIC row's own neighbors/successors consistently with that
    row's own position in the insertion sequence (`build_relationship_diagram`,
    `build_context_lines`) — never for a caller resolving an arbitrary business date (that stays
    `current_gold_state_as_of`/`_entities_as_of`'s `as_of`, unrelated and unchanged)."""
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
    """Batched `entity_id -> its row valid at `as_of`` lookup, one query for the whole list —
    the same `DISTINCT ON` shape as `current_gold_state`/`current_gold_state_as_of`, scoped down
    to a specific set of ids instead of every entity of that type. `as_of=None` means "the latest
    version", matching `current_gold_state`'s own default.

    `before`, when given, takes over entirely (see `_rows_before`) — a `(changed_at, id)`
    cursor, for a caller resolving one specific row's own neighbors, not a business date."""
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
    contracts, split by direction. Reads each spec fresh from the contract's own row, valid at
    `as_of` — never a copy stored on the component itself. See
    `.tmp/improve_timeline_questions_and_linage.md` §3; requires `latest_odcs_spec`'s
    carry-forward (§5) to actually have a non-empty spec to return for most real contracts."""
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
    each one's own row valid at `as_of`. A local, direct read — see `get_successors` for the
    inverse, which is not."""
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
    `component_id`, as of `as_of` (a business date) or `before` (a `(changed_at, id)` cursor —
    see `_rows_before`). There is no `successor_ids` field anywhere: this is derived purely from
    `dependency_ids` read across every component, so it can never drift out of sync with it the
    way a separately maintained inverse column would. See
    `.tmp/improve_timeline_questions_and_linage.md` §2.4."""
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


async def build_relationship_diagram(
    session: AsyncSession, rows: list[GoldEvolution], *, tenant: str = "default"
) -> tuple[str, list[GoldEvolution]] | None:
    """A small Mermaid `flowchart LR` scoped to just the component(s) a chat answer actually
    cited (`rows` here — see `chat`'s own filtering by `ChatCitation`), plus each one's direct
    neighbors: predecessors (`ComponentPayload.dependency_ids`, a local read) and successors
    (`get_successors`, the derived inverse). This is deliberately NOT
    `current_architecture_diagram` scoped down — that reads the whole tenant's live state; this
    is built from the exact rows an answer already resolved, so it can never show a component
    the answer never actually grounded.

    Returns `None` when there is nothing to show: no component among `rows` (e.g. the answer was
    only about a data contract), or a component with zero dependencies and zero successors.
    Otherwise returns `(mermaid_source, focal_rows)` — `focal_rows` is exactly the cited
    component rows passed in, never a neighbor, so the caller can build "View ADR" links from
    the entity the answer was actually about.

    A historical focal row (`valid_to is not None` — i.e. superseded, see
    `_answer_specific_version_question`) has its neighbors resolved with a `(changed_at, id)`
    cursor (`_rows_before`) pinned to THAT row's own insertion moment, never "whatever is
    current now". A real, reported bug: pinning an answer to an old version still showed today's
    full neighbor set (a dependency added long after that version), because `get_successors`
    used to be called unscoped. `ingestion_date` alone cannot fix this — several versions of the
    same entity can share one calendar day (several ADRs uploaded together) — only the
    insertion-order cursor can. A still-current focal row (`valid_to is None`) keeps reading
    today's true state, exactly as before: it would be its own regression to show a stale
    neighbor for a component that just happens not to have changed as recently as its neighbors.

    Reuses `current_architecture_diagram`'s exact style conventions (`classDef`, backtick
    multi-line labels, no `click` directive) — see that function's own docstring for the real
    `st.mermaid_chart` rendering bug those choices avoid. A second `classDef` (`focalNode`)
    highlights the cited component(s) so a reader can tell "what this answer is about" from
    "context neighbor" at a glance."""
    focal_rows = [row for row in rows if row.entity_type == "component"]
    if not focal_rows:
        return None

    focal_ids = {row.entity_id for row in focal_rows}
    # (from_id, to_id) meaning from_id depends on to_id — the same direction
    # `current_architecture_diagram`'s `component --> dependency` edges use.
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

# This many candidates get fetched from a ranking before it is cut down to `k`. Three unrelated
# things all need this same extra depth, so they share one constant:
#
# 1. `mode="hybrid"` fuses TWO rankings (vector, lexical) with RRF. A row that ranks, say, 7th by
#    embedding similarity and 2nd by lexical match should still be able to win a fused top-5 over
#    two rows that only rank in the top 5 of one signal and nowhere in the other. Fusing two
#    lists already cut to `k` would never give a moderate-vector/strong-lexical row that chance.
# 2. `dedupe=True` (the default, either mode) collapses several VERSIONS of the SAME entity down
#    to one — its actual latest version, see `_dedupe_ids_by_entity` — before cutting to `k`.
#    Several near-identical consecutive-version narratives can otherwise occupy most of a plain
#    top-k, crowding out a genuinely different, relevant entity (`.tmp/advanced_techniques.md`
#    §2, and the same problem the estimator project's own `dedupe` flag measured and fixed).
# 3. `rerank=True` (opt-in, off by default) repunctuates the deduped-but-not-yet-cut candidates
#    with a real cross-encoder before cutting to `k` — see `_rerank_ids` and
#    `.tmp/advanced_techniques.md` §3. It needs the same wide pool to have anything worth
#    repunctuating; reranking a list already cut to `k` could never promote a candidate ranked
#    just outside it.
RECALL_POOL_SIZE = 30

# `RRF_K_CONSTANT`, `_reciprocal_rank_fusion`, `_top_k_ids_by_vector`, and
# `_top_k_ids_by_lexical_rank` (hybrid search); `_dedupe_ids_by_entity` (deduplication); and
# `_rerank_ids` (reranking) now live in `agents/stages/gold/retrieval/`, one file per
# technique — imported above. This function is their orchestrator, not their implementation.


async def embed_question(text: str) -> list[float]:
    """`ingestion.embedder.embed` is a blocking, batched call. This wraps it in
    `asyncio.to_thread`, the same way `persist_entity_version` already does, applied here to one
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
    (`_reciprocal_rank_fusion`) — see `.tmp/advanced_techniques.md` §1 for why: an embedding can
    blur an exact component name, acronym, or ODCS field name that a plain keyword match finds
    immediately, at close to zero extra cost (the GIN index, versus the embedding API/model call
    already being paid for `vector`). `mode="hybrid"` requires `question_text` — the original
    question, not `vector`'s embedding of it, since the lexical branch does its own tokenization,
    never the embedding.

    `dedupe=True` (the default) collapses several versions of the SAME entity down to one — its
    actual latest version, see `RECALL_POOL_SIZE`'s own comment, `_dedupe_ids_by_entity`, and
    `.tmp/advanced_techniques.md` §2. `dedupe=False` reproduces the exact pre-deduplication
    behavior (a plain top-`k` cut, no wider recall fetched first) — kept for tests and callers
    that need to see the raw, undeduplicated ranking.

    `rerank=True` (opt-in, off by default) repunctuates the deduped-but-not-yet-cut candidates
    with a real local cross-encoder (`ingestion.reranker.score_candidates`) before cutting to
    `k` — see `_rerank_ids` and `.tmp/advanced_techniques.md` §3. A cross-encoder reads
    `(question, narrative)` jointly, one forward pass per candidate, instead of comparing two
    separately-computed vectors — this can promote a candidate a plain vector/hybrid ranking
    left just outside `k`, at the cost of one extra model call per candidate. `rerank=True`
    requires `question_text`, in every mode, including `mode="vector"` — the cross-encoder
    always needs the original question text, never `vector`'s embedding of it.

    `expand=True` (opt-in, off by default) asks `expand_question` for a few reformulations of
    `question_text`, searches with each of them too (embedding and, in `mode="hybrid"`, the
    lexical search as well), and fuses every one of those extra rankings into the same
    `_reciprocal_rank_fusion` pool this function already uses for hybrid search — see
    `retrieval/query_expansion.py` and `.tmp/advanced_techniques.md` §4. This is the one gap
    hybrid search alone cannot close: a synonym that shares no word at all with the canonical
    name ("el módulo de pagos" vs. "Payments Gateway") never matches lexically, and may not
    embed close enough either — a reformulation that happens to land closer to the canonical
    wording rescues it. `expand=True` requires `question_text`, in every mode, for the same
    reason `rerank=True` does."""
    if rerank and not question_text:
        raise ValueError("rerank=True requires question_text (the cross-encoder needs the raw question)")
    if expand and not question_text:
        raise ValueError("expand=True requires question_text (needed to generate reformulations)")

    # Hybrid search and query expansion both fuse more than one ranking together, so both need
    # the same wide recall depth as dedup/rerank — a shallow `k`-sized fetch from any one of
    # them would starve the fusion of candidates the OTHER rankings might have promoted.
    fuse_deep = mode == "hybrid" or expand
    recall = RECALL_POOL_SIZE if (dedupe or rerank or fuse_deep) else k

    rankings: list[list[int]] = []
    if mode == "vector":
        rankings.append(await _top_k_ids_by_vector(session, vector, recall, source_component, max_distance, tenant))
    elif mode == "hybrid":
        if not question_text:
            raise ValueError("mode='hybrid' requires question_text (the lexical ranking needs the raw question)")
        # Sequential, not `asyncio.gather` — a single `AsyncSession` cannot run two queries
        # concurrently over one DBAPI connection. Doing so raised a real
        # `sqlalchemy.exc.IllegalStateChangeError` here, confirmed against the actual running
        # app, not just a theoretical concern: two coroutines sharing `session` inside
        # `gather` both took the connection into an "in progress" state at once, and closing
        # the session afterward hit that half-finished state.
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

    # `rerank=True` needs the deduped pool BEFORE it is cut to `k` — reranking a list already
    # cut to `k` could never promote a candidate ranked just outside it. So dedup is asked for
    # up to `RECALL_POOL_SIZE` entities here, not `k`, whenever a rerank pass still follows.
    dedupe_limit = RECALL_POOL_SIZE if rerank else k
    ids = await _dedupe_ids_by_entity(session, ids, dedupe_limit) if dedupe else ids[:dedupe_limit]

    ids = await _rerank_ids(session, question_text, ids, k) if rerank else ids[:k]

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


async def _resolve_entity_names(
    session: AsyncSession, tenant: str, ids_by_type: dict[str, set[str]]
) -> dict[tuple[str, str], str]:
    """Batched `entity_id -> canonical_name` lookup for `answer_question`'s payload
    enrichment below. A component's `dependency_ids`/`contract_ids` and a data contract's
    `producer_id`/`consumer_id` (`ComponentPayload`/`DataContractPayload`) only ever store
    resolved ids, never names — showing a raw ULID to the answering LLM would be useless. This
    picks each id's latest version's `canonical_name`, one query per entity_type, not one
    query per id."""
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
    `answer_question`'s context line — the detail a "what does X depend on" / "what does this
    contract cover" / "what depends on X" question needs, and that `row.narrative` alone does
    not reliably restate. Returns `""` when the payload has nothing to add (e.g. an
    architecture-entity row, or a component with no known dependencies/contracts/successors).

    `successors`, when given, maps a component's own `entity_id` to the canonical names of
    every OTHER component that currently depends on it (`get_successors`'s own return, reduced
    to names by the caller). There is no `successor_ids` field on the payload itself to read
    here — unlike `dependency_ids` (this component's own local, direct field), successors are
    computed by the caller by scanning every other component, never stored — see
    `.tmp/improve_timeline_questions_and_linage.md` §2.4."""
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
            # Rows persisted before the input/output split (§2.2, §6) only ever have the old,
            # undifferentiated `contract_ids` — fall back to it so they do not go silent.
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

# Real bug this closes: `_payload_detail`'s data_contract branch only ever showed schema FIELD
# NAMES, never the contract's actual specification — so "show me the spec of X" got an honest
# "the retrieved facts don't provide it" even when `odcs_spec` was fully populated. This gates a
# much bigger block (the entire spec) behind an explicit ask, so an ordinary question about a
# contract never pays for it in context tokens it did not request.
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
    """Returns `True` if `question` reads as asking for a data contract's complete
    specification, in either English or Spanish — see `_FULL_SPEC_MARKERS`. Gates
    `build_context_lines`'s per-row full-spec block, the same way `is_evolution_question` gates
    the full-history path."""
    lowered = question.lower()
    return any(marker in lowered for marker in _FULL_SPEC_MARKERS)


def _full_odcs_spec_block(row: GoldEvolution) -> str:
    """The entire `odcs_spec` for one data_contract row, pretty-printed. `""` for any other
    entity_type, or a contract with no spec captured yet (`parse_odcs_spec`'s own `{}`
    fallback — see `.tmp/improve_timeline_questions_and_linage.md` §5's carry-forward fix for
    why that should now be rare for a contract with any real history)."""
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
    """One line per retrieved row, exactly as `answer_question` renders it into its prompt —
    narrative plus the resolved `_payload_detail` clause (dependencies/successors/contracts),
    plus a data contract's full `odcs_spec` when `question` asks for it (`wants_full_spec`).
    Pulled out as its own function so a caller other than `answer_question` (namely
    `agents/stages/gold/testing/chat_eval/test_chat_eval.py`'s RAGAS scoring) can score
    faithfulness/context_recall against what the model actually saw, not just `row.narrative` —
    passing bare narratives as RAGAS `contexts` understates faithfulness for any answer that
    correctly used payload-derived facts (dependencies, contract direction, successors) the
    judge was never shown. See `.tmp/improve_timeline_questions_and_linage.md` §7.

    `question` is optional (`None` skips the full-spec block entirely) so existing callers that
    only ever wanted the old behavior do not need to change."""
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
    # One `get_successors` query per retrieved component — cheap at this scale (`rows` is a
    # bounded top-k, not the whole tenant). This is the only way to surface successors at all:
    # unlike `dependency_ids`, there is no stored field to just read off the payload.
    #
    # A historical row (`valid_to is not None` — pinned by `_answer_specific_version_question`)
    # is cursored to its own extraction batch's end (`_batch_cursor`), never "current": leaving
    # this unscoped was a real, reported bug — a v1-pinned answer still listed today's full
    # successor set, including a dependency added long after v1. `ingestion_date` alone cannot
    # fix this — several versions of the same entity can share one calendar day. A still-current
    # row (`valid_to is None`, the ordinary top-k case) keeps reading today's true state.
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
    plus `citations` naming exactly which rows it drew from. This is the RAG-consumer step
    that top-k retrieval alone does not cover.

    Every citation is checked against `rows` by `_verify_citations` before this returns — the
    model can cite a fact, but it cannot make that citation stick unless it names a row that
    was genuinely retrieved. This is why the context below includes each row's raw
    `entity_id`: without it, the model would have nothing correct to cite even when it wanted
    to.

    `latest` comes from `latest_versions` and is optional. It tags each row as `[latest version]`
    or `[superseded — latest is version N]`, in the context the LLM sees. This stops a question
    about current state from getting answered using a row that ranked close by embedding
    similarity but has since been superseded. Without `latest`, every row is presented the same
    way. That is what happens today when a caller does not pass it.

    `session` is used only to resolve each row's `payload` ids (a component's dependencies and
    contracts, a data contract's producer/consumer/schema fields — see `_payload_detail`) into
    names the LLM can actually read; `rows` themselves are not re-fetched. Before this, only
    `narrative` ever reached this prompt, so a question genuinely asking for a component's
    dependencies or a contract's schema had no way to be answered even when that exact data was
    already sitting in `payload` — this closes that gap.

    `history`, when given, is the most recent turns of this same chat session (see
    `_format_history_block` and `.tmp/advanced_techniques.md` §8). It is rendered as its own
    labeled block, never merged into `question` itself, so the model always sees one clean,
    unambiguous line to answer. This is what lets a follow-up like "and who approved it?"
    resolve "it" to whatever entity the previous turn was actually about, instead of being
    answered as a brand-new, context-free question — the prompt below is explicit that history
    may only be used to resolve what the question REFERS to, never as a source of facts on its
    own; every fact in the answer must still come from `rows`."""
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
    # `valid_to` (`.tmp/improve_timeline_questions_and_linage.md` §2.1) is what lets an
    # evolution answer say how long a version actually lasted, not just when it started —
    # "in effect Jan-Mar 2026" instead of only "asserted on 2026-01-15".
    validity = "still in effect" if row.valid_to is None else f"in effect until {row.valid_to.isoformat()}"
    return (
        f"- [{row.entity_type}:{row.entity_id}] Version {row.version}, {row.ingestion_date.isoformat()}, "
        f"by {author}, from ADR {row.source_component} v{row.source_adr_version} "
        f"(operation={row.operation}, {validity}): {row.narrative}"
    )


async def answer_evolution_question(
    question: str, canonical_name: str, rows: list[GoldEvolution], history: list[tuple[str, str]] | None = None
) -> GroundedAnswer:
    """Drafts a structured answer to a "how has X evolved over time" question, from EVERY
    version `entity_history` returned for one entity, oldest first — plain-text `answer` plus
    `citations` naming exactly which version(s) it drew from, checked against `rows` by
    `_verify_citations` the same way `answer_question` does. This is deliberately a separate
    function from `answer_question`, not a shared one with an extra flag: that function answers
    from a top-k, possibly cross-entity, possibly incomplete row set, and tells the LLM to
    prefer the [latest version] tag for current-state questions. This function's whole point is
    the opposite — narrate the complete, ordered sequence of changes, not settle on one current
    answer.

    Each step's line names its entity, version, event-time date (`ingestion_date`, when the
    change was asserted, not `changed_at`), author (`authored_by`, empty for a CLI/script/test
    run with no real user — see `GoldEvolution.authored_by`'s own column comment), the exact
    ADR it came from (`source_component`/`source_adr_version`), operation, and narrative.
    The prompt asks for a chronological prose account: who introduced the entity, when, and
    why, then each later step and its own reason, in the style of "X introduced this on Y for
    Z, then evolved on V because...". `rows` empty means the entity was never found; the caller
    is expected to have already checked that via `find_entity_by_name_in_text` before calling
    this, but this function still degrades safely instead of sending an empty prompt.

    `history`, when given, is the most recent turns of this same chat session — see
    `_format_history_block` and `.tmp/advanced_techniques.md` §8. It is what let the CALLER
    even resolve `canonical_name` in the first place, for a follow-up like "how has it evolved
    over time?" that never names the entity itself (see `app.routers.frontend.chat`'s own
    contextualized-text step). It is passed here too, purely so the narrated answer can match
    the conversation's own phrasing (e.g. "it" instead of repeating the full entity name every
    sentence) — never as a source of facts; every fact must still come from `rows`."""
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
    "build_context_lines",
    "build_relationship_diagram",
    "classify_contract_directions",
    "content_hash",
    "contracts_with_real_changes",
    "current_architecture_diagram",
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
    "parse_odcs_spec",
    "persist_entity_version",
    "resolve_and_alias",
    "resolve_entity_id",
    "resolve_entity_id_for_lookup",
    "retrieve_with_correction",
    "top_k_gold_evolution",
    "wants_full_spec",
]
