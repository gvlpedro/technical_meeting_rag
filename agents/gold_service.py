"""Reusable Gold logic — callable directly (a future backfill script, unit tests) without
needing the full LangGraph state shape. Mirrors `agents/service.py`'s own split relative to
`agents/graph.py`: the three Gold nodes in `agents/graph.py` (`extract_gold_facts`,
`resolve_gold_identity`, `persist_gold_evolution`) are thin wrappers calling the functions
here.

Gold's per-ADR reconciliation runs inside Silver's own graph run, not as a separately triggered
pipeline watching `silver_documents` for changes — see `.tmp/gold_process_v5.md` §1-2 for why.
Every function below is still a plain function taking explicit arguments (no graph `state`),
so `.tmp/gold_process_v5.md` §4's standalone rebuild path (a future `scripts/backfill_gold.py`
replaying Gold over existing `silver_documents` rows without re-running Silver's ACB loop) can
call the exact same functions the graph does.
"""

import asyncio
import hashlib
import json
from collections.abc import Sequence
from datetime import date
from uuid import uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from agents.prompts import build_gold_extraction_prompt
from agents.schemas import (
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

# Same criterion `resolve_entity_id` uses either side of an ambiguous match — a lower value
# would start resolving genuinely-different names to the same entity_id (e.g. "Order Service"
# vs. "Orders API" on a bad day); a higher one would mint duplicate entity_ids for trivial
# spelling/casing drift `gold_aliases` exists specifically to absorb. Not tuned against a real
# corpus yet — a candidate for revisiting once real transcripts exercise this path.
FUZZY_MATCH_THRESHOLD = 0.6


def content_hash(content: str) -> str:
    """Shared by `_entity_hash` below and `agents.graph._persist_document_version` — one
    definition of "how we turn a string into our version-hash" for both Silver's whole-document
    hash and Gold's per-entity hash, even though they hash different-shaped things (v6 §4)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _entity_hash(operation: GoldOperation, narrative: str, payload: dict) -> str:
    """Same hash-compare-then-bump mechanism as `content_hash`/`agents.graph.
    _persist_document_version`, scoped to one entity's own fields instead of a whole document —
    see `GoldEvolution.entity_hash`'s column comment for why it isn't named `content_hash`.
    `payload` is serialized with `sort_keys=True` so two logically-identical payloads with keys
    inserted in a different order still hash identical — a real risk here since `payload`
    round-trips through Pydantic `.model_dump()` and plain dict construction in different call
    sites. `sort_keys=True` only normalizes dict key order, NOT list element order —
    `payload["dependency_ids"]`/`["contract_ids"]`/etc. must already be sorted by the caller
    before this is called (see `agents.graph.persist_gold_evolution` and
    `scripts.backfill_gold._backfill_one`), or two logically-identical extractions whose LLM
    call happened to list the same names in a different order would hash differently and
    trigger a spurious version bump."""
    canonical_payload = json.dumps(payload, sort_keys=True)
    return content_hash(f"{operation}\n{narrative}\n{canonical_payload}")


async def extract_gold_facts_for_source(
    adr_content: str, mentioned_components: list[dict], mentioned_data_contracts: list[dict]
) -> GoldExtractionResult:
    """One structured-extraction LLM call, `temperature=0` — a deterministic extraction task,
    not creative writing, same rationale as `generate_architecture_questions_for_batch`'s own
    `temperature=0, reasoning_effort="none"` (`agents/service.py`). `reasoning_effort="none"`
    is required alongside it for the same documented reason: a reasoning-locked model (e.g.
    gpt-5.6-terra) otherwise rejects any temperature but 1. Anthropic's reasoning-locked models
    (e.g. claude-opus-5) have no equivalent escape hatch and hard-require temperature=1 — if
    `settings.llm_fallback_order` ever falls through to Anthropic for this call, it fails
    instead of gracefully falling back. Same known, accepted-for-now risk as the call this
    mirrors; not fixed here (would mean either reordering the fallback for every deterministic
    call in this codebase, or teaching the router itself to drop `temperature`/
    `reasoning_effort` per-provider — a bigger change than this pass's scope)."""
    messages = build_gold_extraction_prompt(adr_content, mentioned_components, mentioned_data_contracts)
    response = await router.complete(
        messages, response_format=GoldExtractionResult, temperature=0, reasoning_effort="none"
    )
    return GoldExtractionResult.model_validate(load_json_response(response.choices[0].message.content))


async def already_extracted(session: AsyncSession, source_component: str, source_adr_version: int) -> bool:
    """True if Gold has already written at least one `gold_evolution` row for this exact
    `(source_component, source_adr_version)` pair — this specific version of this ADR was
    already reconciled, nothing to re-extract. Simpler than tracking a separate "last
    processed content_hash" column (an earlier draft of this idea, `.tmp/gold_process_v5.md`
    §3): `source_adr_version` is itself already the output of Silver's own hash-based
    versioning, so an unchanged ADR never gets a new version to begin with — checking whether
    THIS version was processed is equivalent and needs no extra column."""
    result = await session.execute(
        select(GoldEvolution.id)
        .where(
            GoldEvolution.source_component == source_component,
            GoldEvolution.source_adr_version == source_adr_version,
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def resolve_entity_id(session: AsyncSession, entity_type: GoldEntityType, name: str) -> str:
    """Exact match on `gold_aliases.alias` -> `pg_trgm` fuzzy match above
    `FUZZY_MATCH_THRESHOLD` (best similarity wins) -> mint a new `entity_id` (`uuid4`, as a
    string — `gold_evolution`/`gold_aliases` never treat it as anything but an opaque string,
    same "owns nothing, purely a lookup" principle `gold_process.md` §3 states for
    `gold_aliases`). Does not insert the alias row itself — see `ensure_alias`: a name that
    resolves to an EXISTING entity_id via fuzzy match still needs its own alias row inserted
    (a new variant of a known entity), which is a different operation from minting."""
    exact = (
        await session.execute(
            select(GoldAlias.entity_id).where(GoldAlias.entity_type == entity_type, GoldAlias.alias == name)
        )
    ).scalars().first()
    if exact is not None:
        return exact

    fuzzy = (
        await session.execute(
            select(GoldAlias.entity_id)
            .where(
                GoldAlias.entity_type == entity_type,
                func.similarity(GoldAlias.alias, name) > FUZZY_MATCH_THRESHOLD,
            )
            .order_by(func.similarity(GoldAlias.alias, name).desc())
            .limit(1)
        )
    ).scalars().first()
    if fuzzy is not None:
        return fuzzy

    return str(uuid4())


async def ensure_alias(
    session: AsyncSession,
    entity_type: GoldEntityType,
    entity_id: str,
    alias: str,
    source_component: str,
    source_adr_version: int,
) -> None:
    """Inserts this `(entity_type, entity_id, alias)` row if it doesn't already exist —
    `ON CONFLICT DO NOTHING`, same idempotent-insert convention the rest of this pipeline uses
    for anything that can legitimately re-run over the same source (e.g.
    `agents.graph._persist_document_version`'s overwrite branch). Called both for a brand-new
    entity_id's first alias and for an existing entity_id seen under a new name variant — both
    cases just insert-if-absent, no different logic needed for either."""
    stmt = (
        insert(GoldAlias)
        .values(
            entity_type=entity_type,
            entity_id=entity_id,
            alias=alias,
            source_component=source_component,
            source_adr_version=source_adr_version,
        )
        .on_conflict_do_nothing(index_elements=["entity_type", "entity_id", "alias"])
    )
    await session.execute(stmt)


async def resolve_and_alias(
    session: AsyncSession,
    entity_type: GoldEntityType,
    name: str,
    name_to_id: dict[str, str],
    source_component: str,
    source_adr_version: int,
) -> str:
    """`resolve_entity_id` + `ensure_alias` composed, memoized against the caller's own
    source-scoped `name_to_id` map so a name already resolved earlier in the same batch is a
    dict lookup, not a second round trip. The one identity-resolution step
    `agents.graph.resolve_gold_identity` and `scripts.backfill_gold._backfill_one` both need —
    previously a nested closure in the former and a near-identical module-level function in the
    latter, now this single definition either calls."""
    if name in name_to_id:
        return name_to_id[name]
    entity_id = await resolve_entity_id(session, entity_type, name)
    name_to_id[name] = entity_id
    await ensure_alias(session, entity_type, entity_id, name, source_component, source_adr_version)
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
) -> int | None:
    """Hash-compare-then-bump against the latest `gold_evolution` row for this
    `(entity_type, entity_id)` — a query (`ORDER BY version DESC LIMIT 1`), not a materialized
    cache lookup, since `gold_current_state` is cut for this pass (`.tmp/optmizaciones.md` §1).
    Same mechanism as `agents.graph._persist_document_version`, scoped to one entity: identical
    `entity_hash` -> no-op, returns `None`; different (or no prior row at all) -> insert
    `version + 1` (or `version = 1`), embedding `narrative` via the same
    `ingestion/embedder.py` singleton Bronze/Silver already use — no second embedding stack.

    Caller's responsibility, not this function's: never call this for an `operation ==
    "unknown"` entity — see `agents.graph.persist_gold_evolution`'s skip before it ever reaches
    here. `GoldOperation` types `operation` as the union of all three per-`entity_type` literals
    (still including `"unknown"` at the type level, since Python can't easily express "minus one
    value"), but this function doesn't itself check that `operation` is the right *subset* for
    the given `entity_type` — that pairing is the caller's responsibility."""
    entity_hash = _entity_hash(operation, narrative, payload)
    latest = (
        await session.execute(
            select(GoldEvolution)
            .where(GoldEvolution.entity_type == entity_type, GoldEvolution.entity_id == entity_id)
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
        )
    )
    return version


async def current_gold_state(
    session: AsyncSession, entity_type: GoldEntityType | None = None
) -> Sequence[GoldEvolution]:
    """The "current state" read path `v4`'s now-cut `gold_current_state` table would have
    served — one row per `(entity_type, entity_id)`, always the latest version, as a query
    over `gold_evolution` instead of a maintained cache (`.tmp/optmizaciones.md` §1). Postgres
    `DISTINCT ON` needs its own `ORDER BY` prefix matching the `DISTINCT ON` columns before the
    tie-breaking `version DESC`, so this is raw column ordering, not something the ORM's
    `.distinct()` expresses directly."""
    query = select(GoldEvolution).distinct(GoldEvolution.entity_type, GoldEvolution.entity_id)
    if entity_type is not None:
        query = query.where(GoldEvolution.entity_type == entity_type)
    query = query.order_by(GoldEvolution.entity_type, GoldEvolution.entity_id, GoldEvolution.version.desc())
    return (await session.execute(query)).scalars().all()


# --- Top-k retrieval — the RAG-consumer read path `current_gold_state` above doesn't cover
# (that's "latest version per entity"; this is "which versions, across the whole history, are
# semantically closest to a question"). Shared by `scripts/chat_gold.py` (interactive use) and
# `testing_gold_arch_evolution/retrieval.py` (asserted against by that suite) so the retrieval
# mechanism itself has exactly one definition. -----------------------------------------------

# Cosine distance (1 - cosine_similarity) above which a row is dropped from top-k results
# instead of being forced into the answer — plain top-k always returns exactly `k` rows even
# when none of them are actually relevant to the question, which is how a RAG answer ends up
# confidently invented from the least-bad match instead of saying "I don't know." Same caveat
# as `FUZZY_MATCH_THRESHOLD` above: not tuned against a real corpus yet, a starting point to
# revisit once real chat usage gives us distances to calibrate against. `max_distance` is
# opt-in on `top_k_gold_evolution` (default `None`, i.e. unfiltered) precisely so this default
# can be tuned later without silently changing what `testing_gold_arch_evolution`'s already
# real-LLM-verified assertions see.
DEFAULT_MAX_DISTANCE = 0.6


async def embed_question(text: str) -> list[float]:
    """`ingestion.embedder.embed` is a blocking, batched call — same `asyncio.to_thread`
    wrapping `persist_entity_version` already uses, applied here to one question at a time."""
    [vector] = await asyncio.to_thread(embed, [text])
    return vector


async def top_k_gold_evolution(
    session: AsyncSession,
    vector: list[float],
    k: int = 5,
    source_component: str | None = None,
    max_distance: float | None = None,
) -> list[GoldEvolution]:
    """Cosine-distance top-k over ALL versions of `gold_evolution` (not just the latest per
    entity — that's `current_gold_state`'s job), ordered by `.cosine_distance(vector)` so the
    query shape matches `GoldEvolution.embedding`'s HNSW index (`vector_cosine_ops`) and can
    actually use it. `source_component` narrows to one source when given, so unrelated rows
    elsewhere in the database can't win a top-k slot; omitted, it searches the whole table.
    `max_distance`, when given (see `DEFAULT_MAX_DISTANCE`), drops rows past that distance
    instead of always returning `k` regardless of relevance — filtered in SQL, not after the
    fact, so a tight bound also means less work fetched over the wire."""
    distance = GoldEvolution.embedding.cosine_distance(vector)
    query = select(GoldEvolution).order_by(distance).limit(k)
    if source_component is not None:
        query = query.where(GoldEvolution.source_component == source_component)
    if max_distance is not None:
        query = query.where(distance <= max_distance)
    result = await session.execute(query)
    return list(result.scalars().all())


async def latest_versions(session: AsyncSession, rows: Sequence[GoldEvolution]) -> dict[tuple[str, str], int]:
    """The current (latest) version number for each distinct `(entity_type, entity_id)` appearing
    in `rows` — one batched query, not one lookup per row. Lets `answer_question` tell the LLM
    which retrieved rows are still current vs. superseded by a later version that may not have
    made the same top-k (top-k is searched across ALL versions, so an older version's narrative
    can rank closer to the question than its own entity's current one — see `top_k_gold_evolution`'s
    own docstring)."""
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
    """Drafts a plain-text answer to `question` from the retrieved rows — the RAG-consumer step
    top-k retrieval alone doesn't cover. Plain text out, no `response_format`, same convention
    `testing_adr_acb`'s own Actor call uses for its final ADR document: this prompt writes an
    answer directly, not structured JSON.

    `latest` (from `latest_versions`, optional) tags each row as `[latest version]` or
    `[superseded — latest is version N]` in the context the LLM sees, so a question about current
    state doesn't get answered from a row that ranked close by embedding similarity but has since
    been superseded — without it, every row is presented the same way (current behavior when
    callers don't pass it)."""
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


__all__ = [
    "ArchitecturePayload",
    "ComponentPayload",
    "DEFAULT_MAX_DISTANCE",
    "DataContractPayload",
    "FUZZY_MATCH_THRESHOLD",
    "already_extracted",
    "answer_question",
    "content_hash",
    "current_gold_state",
    "embed_question",
    "ensure_alias",
    "extract_gold_facts_for_source",
    "latest_versions",
    "persist_entity_version",
    "resolve_and_alias",
    "resolve_entity_id",
    "top_k_gold_evolution",
]
