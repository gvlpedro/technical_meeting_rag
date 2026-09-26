"""This file holds the structured-output and storage schemas for the Gold stage. It has two
parts: extraction (`GoldExtractionResult`, one LLM call per finished ADR), and the
`gold_evolution.payload` shapes each entity type saves as (`ComponentPayload`,
`DataContractPayload`, `ArchitecturePayload`). See `.tmp/gold_process.md` v4,
`.tmp/gold_process_v5.md`, and `.tmp/refactor_silver_and_gold_process_v6.md` for the design
history behind this stage."""

from typing import Any, Literal

from pydantic import BaseModel

from agents.shared import ComponentStatus, ContractAction

# `gold_evolution.entity_type`.
GoldEntityType = Literal["component", "data_contract", "architecture"]

# This is `"changed"` or `"unchanged"`. `ComponentStatus` and `ContractAction` do not fit the
# architecture-as-a-whole entity, because there is no "new" or "removed" architecture. So this
# entity gets its own, smaller literal, instead of reusing one of the other two types for a
# meaning it was not designed for (v6 §3).
ArchitectureChangeType = Literal["changed", "unchanged"]
# This is `gold_evolution.operation`. It is `ComponentStatus` when `entity_type == "component"`,
# `ContractAction` when `entity_type == "data_contract"`, and `ArchitectureChangeType` when
# `entity_type == "architecture"` (v6 §3). This is one combined type because
# `agents.stages.gold.service.persist_entity_version` and `_entity_hash` save all three shapes
# through the same column. They cannot pick a single one of the three literals ahead of time, so
# they need one combined type instead of falling back to a bare `str`.
#
# This type still includes "unknown" at the type level, inherited from `ComponentStatus` and
# `ContractAction`. The actual guarantee that "unknown" never reaches this column at runtime
# lives in `agents.graph.persist_gold_evolution`'s skip step, not here.
GoldOperation = ComponentStatus | ContractAction | ArchitectureChangeType


class ComponentPayload(BaseModel):
    """This is the `gold_evolution.payload` shape for `entity_type == "component"`. It holds
    only exact, structural data: `entity_id` references, already resolved by
    `resolve_gold_identity`. It never holds raw names. This shape is small and fully typed,
    unlike `DataContractPayload` below.

    `dependency_ids` doubles as this component's predecessors — components it depends on,
    written locally from this same ADR's own extraction. There is no `successor_ids` field:
    successors are the inverse of `dependency_ids` read across every OTHER component, computed
    on demand by `agents.stages.gold.service.get_successors`, never stored — see
    `.tmp/improve_timeline_questions_and_linage.md` §2.4 for why a stored inverse would drift.

    `input_contract_ids`/`output_contract_ids` split `contract_ids` by direction (this component
    as consumer vs. producer), computed the same way `contract_ids` always has been — from this
    same ADR's own contract extraction, never a query. `contract_ids` itself stays, unsplit, for
    rows persisted before this split existed (`.tmp/improve_timeline_questions_and_linage.md`
    §2.2, §6)."""

    dependency_ids: list[str] = []
    contract_ids: list[str] = []
    input_contract_ids: list[str] = []
    output_contract_ids: list[str] = []


class DataContractPayload(BaseModel):
    """This is the `gold_evolution.payload` shape for `entity_type == "data_contract"`. It is
    deliberately not fully typed (v6 §5). ODCS keeps growing new fields: quality rules, servers,
    team and ownership info, SLAs. It is already an external, versioned standard. Re-modeling all
    of it in Pydantic would mean keeping a second, parallel copy. That copy would drift out of
    date the moment ODCS adds a field this schema did not expect.

    `producer`/`consumer` are typed here as the raw component names the ADR used, kept for
    display. `producer_id`/`consumer_id` are those same two components' own resolved
    `entity_id`s (see `ComponentPayload`'s own `dependency_ids`/`contract_ids` — this is the
    same "hold the resolved id, not just the name" principle, applied to a contract's producer
    and consumer). This is what lets a caller answer "which components does this data contract
    touch" by a stable id instead of a name string that a rename or a fuzzy-match drift could
    silently break. Empty string means this contract's producer or consumer was never resolved
    — either it predates this field, or (defensively) resolution failed for a name the ADR
    itself gave. `odcs_spec` is stored and returned as an opaque value."""

    producer: str
    consumer: str
    producer_id: str = ""
    consumer_id: str = ""
    odcs_spec: dict[str, Any] = {}


class ArchitecturePayload(BaseModel):
    """This is the `gold_evolution.payload` shape for `entity_type == "architecture"`. It is a
    snapshot of the whole architecture as of this ADR. It is not a dependency graph in its own
    right. There is no dedicated graph database or edge table for it (`gold_process.md` §8).
    `dependencies` is a flat list of component names. It is kept only as extra context next to
    `mermaid_diagram`, not as structured data on its own."""

    mermaid_diagram: str = ""
    components: list[str] = []
    dependencies: list[str] = []


class ExtractedComponent(BaseModel):
    """This is one component as `extract_gold_facts_for_source` reads it out of the finished
    ADR. It is grounded against the architecture stage's own `MentionedComponent`. It is not
    rediscovered from raw Markdown. The ADR prompt already confirmed the lifecycle status. This
    call refines and confirms that status against the final approved text, per entity, and adds
    a narrative meant for embedding.

    `narrative` is prose. It does not just restate the structured fields. See
    `prompts/gold/extraction.jinja` for the rules. `dependency_names` and `contract_names` are
    raw names, written exactly as they appear in the ADR. `resolve_gold_identity` resolves them
    to `entity_id`s afterward. This model never sees an `entity_id` itself."""

    name: str
    status: ComponentStatus
    narrative: str
    dependency_names: list[str] = []
    contract_names: list[str] = []


class ExtractedDataContract(BaseModel):
    """This is one data contract as `extract_gold_facts_for_source` reads it out of the finished
    ADR. It follows the same grounding and narrative rules as `ExtractedComponent`.

    `odcs_spec` is a **JSON-encoded string** here. It is not a nested object like
    `DataContractPayload.odcs_spec`, which it flows into via
    `agents.stages.gold.service.parse_odcs_spec`. This is deliberate. This class is used as an
    OpenAI structured-output `response_format`. OpenAI's strict JSON-schema mode requires every
    *object*-typed field to declare `additionalProperties: false`. That requirement does not work
    with an intentionally open, unvalidated ODCS blob. See `DataContractPayload.odcs_spec`'s own
    docstring for why that field stays untyped at all.

    Before this string field existed, this was a real bug. It tripped every single time on
    OpenAI. It stayed silently hidden whenever the Anthropic fallback happened to be the one that
    actually ran. A plain string field has no such constraint, so it avoids the bug."""

    name: str
    action: ContractAction
    narrative: str
    producer: str
    consumer: str
    odcs_spec: str = "{}"


class ChatCitation(BaseModel):
    """One citation in a chat answer — points back to the exact `gold_evolution` row (by
    entity + version) a claim came from. `_verify_citations` in
    `agents.stages.gold.service` checks each one against the rows actually retrieved for
    that answer, so a citation is evidence, never just an LLM's unverified claim."""

    entity_type: GoldEntityType
    entity_id: str
    version: int


class GroundedAnswer(BaseModel):
    """`answer_question`/`answer_evolution_question`'s structured output: the plain-text
    answer plus which retrieved rows it actually drew from. Replaces a bare string return so
    citations can be checked mechanically instead of trusted at face value."""

    answer: str
    citations: list[ChatCitation] = []


class QuestionExpansion(BaseModel):
    """`expand_question`'s structured output — 2-3 alternative phrasings of the same
    question, used to widen retrieval to a synonym the canonical wording never used (e.g.
    "el módulo de pagos" vs. the canonical "Payments Gateway"). Never includes the original
    question itself — the caller already has that and decides how to combine them."""

    reformulations: list[str] = []


class GoldExtractionResult(BaseModel):
    """This is the output shape of `extract_gold_facts_for_source`'s one structured-extraction
    LLM call per source (`gold_process.md` §4). That call is folded into Silver's own graph run,
    per `gold_process_v5.md` §2. This is exactly the shape that `resolve_gold_identity` and
    `persist_gold_evolution` consume next.

    An `ExtractedComponent` or `ExtractedDataContract` with status or action `"unknown"` is valid
    output here. That means the extraction genuinely could not resolve it. But such a row is
    never saved to `gold_evolution`. See the caller of
    `agents.stages.gold.service.persist_entity_version`."""

    components: list[ExtractedComponent] = []
    contracts: list[ExtractedDataContract] = []
    architecture_change: ArchitectureChangeType
    architecture_narrative: str
    mermaid_diagram: str = ""
