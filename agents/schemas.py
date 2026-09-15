"""Structured-output schemas for the LLM calls that need one.

`synthesize_document` deliberately has no schema here — it writes the clarified
document directly as Markdown (`doc/silver_process.md` §3 node 5: "no intermediate
structured JSON"), so its response is used as raw text, not parsed against a model.
"""

from typing import Any, Literal

from pydantic import BaseModel

ComponentStatus = Literal["new", "modified", "removed", "unchanged", "unknown"]
# `forward-update` (backward-compatible: additive/optional) vs `break-change` (removes/renames/
# tightens a required field) — see prompts/data_contract_questions.jinja PHASE 4 for the
# criterion. KNOWN LIMITATION: unlike `mentioned_components` (verified verbatim against the
# transcript by `_ungrounded_component_names`), this classification has no mechanical check at
# all — it's whatever the LLM concludes from prompt-following alone. A subtly-breaking change
# that reads like "just adding detail" in the transcript can be misclassified `forward-update`
# with nothing downstream to catch it before it ships as an authoritative ODCS spec. Not fixed
# here — would need a real schema-diff mechanism, out of scope for this pass.
ContractAction = Literal[
    "new", "forward-update", "break-change", "unchanged", "deprecated", "removed", "unknown"
]
QuestionScope = Literal[
    "metadata", "component", "architecture", "data_contract", "adr", "change_impact", "migration"
]
# `gold_evolution.entity_type` — a subset of QuestionScope, imported rather than redeclared
# (refactor v6 §3: one definition both layers import, so a fourth entity type is added once).
GoldEntityType = Literal["component", "data_contract", "architecture"]


class MentionedComponent(BaseModel):
    """One entry of `ArchitectureQuestionListResult.mentioned_components` — a component the
    transcript names, plus the Actor's own read of its lifecycle status relative to
    `KNOWN_ARCHITECTURE` (see prompts/architecture_questions.jinja, COMPONENT
    STATUS)."""

    name: str
    status: ComponentStatus


class MentionedDataContract(BaseModel):
    """One entry of `ArchitectureQuestionListResult.mentioned_data_contracts` — a data contract
    the architecture stage identified (name/producer/consumer/action only, never full schema).
    This is the fixed list `prompts/data_contract_questions.jinja`'s
    `IDENTIFIED_DATA_CONTRACTS` input is built from — the data-contract stage never discovers a
    contract on its own, only this stage does."""

    name: str
    producer: str
    consumer: str
    action: ContractAction


class QuestionItem(BaseModel):
    """One drafted clarification question — carries a stable `id` (see the prompt's
    QUESTION IDENTIFIERS section) so downstream steps (`classify_questions`) can match a
    classification back to its question without relying on exact-text equality."""

    id: str
    scope: QuestionScope
    target: str
    requirement: str
    question: str


class ArchitectureQuestionListResult(BaseModel):
    """`generate_architecture_questions`'s output — see
    prompts/architecture_questions.jinja (question-generation stage 1 of 2: ADR,
    components, and data-contract *identification* — never full ODCS detail, that's
    `DataContractQuestionListResult`'s job). Exactly three top-level fields — the prompt itself
    is explicit that no others are allowed.

    `mentioned_components` is checked mechanically (not just trusted) against the transcript —
    each name must appear in the transcript verbatim, catching an invented or paraphrased
    component name deterministically, without an LLM judge. `mentioned_data_contracts` feeds
    directly into stage 2 as its fixed `IDENTIFIED_DATA_CONTRACTS` list.
    """

    mentioned_components: list[MentionedComponent]
    mentioned_data_contracts: list[MentionedDataContract]
    questions: list[QuestionItem]


class DataContractQuestionListResult(BaseModel):
    """`generate_data_contract_questions`'s output — see
    prompts/data_contract_questions.jinja (question-generation stage 2 of 2: full
    ODCS-completeness questions for each contract `ArchitectureQuestionListResult.
    mentioned_data_contracts` already identified). Every `questions[].scope` is `data_contract`
    and every `questions[].target` must match one of the contracts it was given — checked
    mechanically, same principle as stage 1's component grounding check."""

    questions: list[QuestionItem]


class QuestionClassification(BaseModel):
    # Matched back to its question by `id`, not by re-sending/comparing question text — the
    # classifier only ever needs to echo the id it was given.
    id: str
    answer: str | None = None
    status: Literal["answered", "unknown", "needs_clarification"]


class ClassificationResult(BaseModel):
    classifications: list[QuestionClassification]


class CritiqueClaim(BaseModel):
    # Verbatim substring copied from the drafted document, not paraphrased — this is
    # what lets boss_decide locate and downgrade it in-place later.
    claim: str
    supported: bool
    rationale: str
    severity: Literal["low", "material"]


class CritiqueResult(BaseModel):
    claims: list[CritiqueClaim]


# --- Gold extraction (.tmp/gold_process.md v4, .tmp/gold_process_v5.md,
# .tmp/refactor_silver_and_gold_process_v6.md) -------------------------------------------

# `"changed"|"unchanged"` — ComponentStatus/ContractAction don't apply to the
# architecture-as-a-whole entity (no "new"/"removed" architecture), so it gets its own,
# smaller literal rather than overloading one of the other two with a meaning it wasn't
# designed for (v6 §3).
ArchitectureChangeType = Literal["changed", "unchanged"]
# `gold_evolution.operation` — ComponentStatus for entity_type=="component", ContractAction for
# "data_contract", ArchitectureChangeType for "architecture" (v6 §3). One combined type so
# `agents.gold_service.persist_entity_version`/`_entity_hash`, which persist all three shapes
# through the same column, don't fall back to a bare `str` just because they can't pick a single
# one of the three literals ahead of time. Still includes "unknown" (inherited from
# ComponentStatus/ContractAction) at the type level — the runtime guarantee that "unknown" never
# actually reaches this column lives in `agents.graph.persist_gold_evolution`'s skip, not here.
GoldOperation = ComponentStatus | ContractAction | ArchitectureChangeType


class ComponentPayload(BaseModel):
    """`gold_evolution.payload` shape for `entity_type == "component"` — exact/structural
    data only, `entity_id` references (already resolved by `resolve_gold_identity`), never raw
    names. Small and fully typed, unlike `DataContractPayload` below."""

    dependency_ids: list[str] = []
    contract_ids: list[str] = []


class DataContractPayload(BaseModel):
    """`gold_evolution.payload` shape for `entity_type == "data_contract"`. Deliberately NOT
    fully typed (v6 §5): ODCS keeps growing (quality rules, servers, team/ownership, SLAs) and
    is already an external, versioned standard — re-modeling all of it in Pydantic means
    maintaining a second, parallel copy that drifts the moment ODCS adds a field this schema
    didn't anticipate. Only `producer`/`consumer` are typed, because those are the two fields
    Gold actually filters/joins on; `odcs_spec` is stored and returned opaque."""

    producer: str
    consumer: str
    odcs_spec: dict[str, Any] = {}


class ArchitecturePayload(BaseModel):
    """`gold_evolution.payload` shape for `entity_type == "architecture"` — a snapshot of the
    architecture as a whole as of this ADR, not a dependency graph in its own right (no
    dedicated graph DB/edge table — `gold_process.md` §8); `dependencies` is a flat list of
    component names, kept only as prose-adjacent context for `mermaid_diagram`."""

    mermaid_diagram: str = ""
    components: list[str] = []
    dependencies: list[str] = []


class ExtractedComponent(BaseModel):
    """One component as `extract_gold_facts` reads it out of the finished ADR — grounded
    against `MentionedComponent`, not rediscovered from raw Markdown (the ADR prompt already
    confirmed lifecycle status; this call refines/confirms it against the final approved text,
    per entity, with a narrative meant for embedding).

    `narrative` is prose, not a restatement of the structured fields — see
    `prompts/gold_extraction.jinja`. `dependency_names`/`contract_names` are raw names as
    written in the ADR; `resolve_gold_identity` resolves them to `entity_id`s afterward, this
    model never sees an `entity_id`."""

    name: str
    status: ComponentStatus
    narrative: str
    dependency_names: list[str] = []
    contract_names: list[str] = []


class ExtractedDataContract(BaseModel):
    """One data contract as `extract_gold_facts` reads it out of the finished ADR — same
    grounding/narrative rules as `ExtractedComponent`. `odcs_spec` flows straight into
    `DataContractPayload.odcs_spec`, opaque both here and there."""

    name: str
    action: ContractAction
    narrative: str
    producer: str
    consumer: str
    odcs_spec: dict[str, Any] = {}


class GoldExtractionResult(BaseModel):
    """`extract_gold_facts`'s one structured-extraction LLM call per source
    (`gold_process.md` §4, folded into Silver's own graph run per `gold_process_v5.md` §2) —
    exactly the shape `resolve_gold_identity`/`persist_gold_evolution` consume next. An
    `ExtractedComponent`/`ExtractedDataContract` whose status/action is `"unknown"` is valid
    output here (the extraction genuinely couldn't resolve it) but is never persisted to
    `gold_evolution` — see `agents.gold_service.persist_entity_version`'s caller."""

    components: list[ExtractedComponent] = []
    contracts: list[ExtractedDataContract] = []
    architecture_change: ArchitectureChangeType
    architecture_narrative: str
    mermaid_diagram: str = ""
