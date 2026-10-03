"""Structured-output and storage schemas for the Gold stage: extraction
(`GoldExtractionResult`) and the `gold_evolution.payload` shapes (`ComponentPayload`,
`DataContractPayload`, `ArchitecturePayload`)."""

from typing import Any, Literal

from pydantic import BaseModel

from agents.shared import ComponentStatus, ContractAction

# `gold_evolution.entity_type`.
GoldEntityType = Literal["component", "data_contract", "architecture"]

# The architecture-as-a-whole entity has no "new" or "removed" state, so it gets its own
# literal instead of reusing `ComponentStatus`/`ContractAction`.
ArchitectureChangeType = Literal["changed", "unchanged"]
# `gold_evolution.operation`'s type: `ComponentStatus`, `ContractAction`, or
# `ArchitectureChangeType`, depending on `entity_type`. One combined type because
# `persist_entity_version`/`_entity_hash` save all three shapes through the same column.
#
# This still includes "unknown" at the type level. `agents.graph.persist_gold_evolution` makes
# sure "unknown" never actually reaches this column.
GoldOperation = ComponentStatus | ContractAction | ArchitectureChangeType


class ComponentPayload(BaseModel):
    """The `gold_evolution.payload` shape for `entity_type == "component"`. Holds only
    resolved `entity_id` references, never raw names.

    `dependency_ids` are this component's predecessors. There is no `successor_ids` field:
    successors are the inverse, computed on demand by `get_successors`, never stored, to avoid
    drift.

    `input_contract_ids`/`output_contract_ids` split `contract_ids` by direction (consumer vs.
    producer). `contract_ids` stays, unsplit, for rows persisted before this split existed."""

    dependency_ids: list[str] = []
    contract_ids: list[str] = []
    input_contract_ids: list[str] = []
    output_contract_ids: list[str] = []


class DataContractPayload(BaseModel):
    """The `gold_evolution.payload` shape for `entity_type == "data_contract"`. Not fully
    typed on purpose: ODCS keeps adding fields, and a parallel Pydantic model would drift out
    of date.

    `producer`/`consumer` are the raw component names, for display. `producer_id`/
    `consumer_id` are those components' resolved `entity_id`s, so a caller can find "which
    components touch this contract" by a stable id, not a name a rename could break. Empty
    string means not yet resolved. `odcs_spec` is stored as an opaque value."""

    producer: str
    consumer: str
    producer_id: str = ""
    consumer_id: str = ""
    odcs_spec: dict[str, Any] = {}


class ArchitecturePayload(BaseModel):
    """The `gold_evolution.payload` shape for `entity_type == "architecture"`. A snapshot of
    the whole architecture as of this ADR, not a dependency graph. `dependencies` is a flat
    list of component names, kept as context next to `mermaid_diagram`, not structured data."""

    mermaid_diagram: str = ""
    components: list[str] = []
    dependencies: list[str] = []


class ExtractedComponent(BaseModel):
    """One component as `extract_gold_facts_for_source` reads it from the finished ADR.
    `narrative` is prose, not a restatement of the structured fields. `dependency_names` and
    `contract_names` are raw names, as written in the ADR. `resolve_gold_identity` resolves
    them to `entity_id`s later; this model never sees an `entity_id`."""

    name: str
    status: ComponentStatus
    narrative: str
    dependency_names: list[str] = []
    contract_names: list[str] = []


class ExtractedDataContract(BaseModel):
    """One data contract as `extract_gold_facts_for_source` reads it from the finished ADR.
    Same grounding and narrative rules as `ExtractedComponent`.

    `odcs_spec` is a JSON-encoded string, not a nested object, because OpenAI's strict
    structured-output mode requires `additionalProperties: false` on every object field. That
    does not work with an open ODCS blob. A nested object here was a real bug: it failed on
    OpenAI every time, and stayed hidden when the Anthropic fallback ran instead."""

    name: str
    action: ContractAction
    narrative: str
    producer: str
    consumer: str
    odcs_spec: str = "{}"


class ChatCitation(BaseModel):
    """One citation in a chat answer: points to the exact `gold_evolution` row (entity +
    version) a claim came from. `_verify_citations` checks it against the retrieved rows."""

    entity_type: GoldEntityType
    entity_id: str
    version: int


class GroundedAnswer(BaseModel):
    """`answer_question`/`answer_evolution_question`'s structured output: the answer plus
    which retrieved rows it used. Lets `_verify_citations` check them mechanically."""

    answer: str
    citations: list[ChatCitation] = []


class QuestionExpansion(BaseModel):
    """`expand_question`'s structured output: 2-3 alternative phrasings of the question, used
    to widen retrieval. Never includes the original question itself."""

    reformulations: list[str] = []


class GoldExtractionResult(BaseModel):
    """Output of `extract_gold_facts_for_source`'s one structured-extraction LLM call per
    source. `resolve_gold_identity` and `persist_gold_evolution` consume this next.

    A component or contract with status/action `"unknown"` is valid here, but is never saved
    to `gold_evolution`."""

    components: list[ExtractedComponent] = []
    contracts: list[ExtractedDataContract] = []
    architecture_change: ArchitectureChangeType
    architecture_narrative: str
    mermaid_diagram: str = ""
