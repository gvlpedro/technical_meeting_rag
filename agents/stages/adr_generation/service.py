"""Callable helpers for the ADR-generation stage: pull a previous ADR's diagrams for grounding,
strip display styling before reuse, rebuild §3's node coloring from §4's table, and back up
the prompt's "no placeholders" rule mechanically."""

import re

from agents.shared import _markdown_section, markdown_heading_pattern


def strip_diagram_colors(diagram: str) -> str:
    """Drops every `classDef` and `class` styling line from a Mermaid diagram. A color class
    marks what changed in a diagram; it must never survive into a later document as still
    true. Used by `previous_target_architecture_diagram` and
    `own_previous_architecture_diagram` below, and by `agents.graph.synthesize_document` on
    Gold's own diagram, to keep "## 2. Previous Architecture" a plain, colorless snapshot."""
    lines = [line for line in diagram.splitlines() if not line.strip().startswith(("classDef", "class "))]
    return "\n".join(lines).strip()


# Matches a node declaration under either label style: Gold's backtick style
# (`n0["`Frontend v1\n*...*`"]`) or plain Mermaid brackets (`CheckoutService[Checkout Service]`).
_NODE_DECLARATION = re.compile(r"(\w+)\[(.*?)\]", re.DOTALL)
# One row of §4's table: `| Component | **STATUS** | Description |`. Requires the bold
# `**...**` marker so an unrelated table row is never mistaken for a component row.
_AFFECTED_COMPONENT_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*\*\*(NEW|MODIFIED|REMOVED)\*\*\s*\|", re.MULTILINE)
_STATUS_TO_CLASS = {"NEW": "nodeNew", "MODIFIED": "nodeModified", "REMOVED": "nodeRemoved"}
# Order generator.jinja's own example uses — kept for a readable diff, not because Mermaid cares.
_CLASS_ORDER = ("nodeNew", "nodeModified", "nodeRemoved")
# The fixed palette generator.jinja's DIAGRAM COLOR CODING section specifies, verbatim.
_STANDARD_CLASSDEFS = {
    "nodeNew": "classDef nodeNew fill:#34d399,stroke:#047857,color:#022c22",
    "nodeModified": "classDef nodeModified fill:#fb923c,stroke:#c2410c,color:#431407",
    "nodeRemoved": "classDef nodeRemoved fill:#f87171,stroke:#b91c1c,color:#450a0a,stroke-dasharray: 5 5",
}


def _node_id_sort_key(node_id: str) -> tuple[int, str]:
    """Sorts `n0` before `n1` before `n10` numerically. Falls back to alphabetical order for
    a non-numeric id."""
    match = re.search(r"\d+$", node_id)
    return (int(match.group()), node_id) if match else (-1, node_id)


def _node_name_keys(raw_bracket_content: str) -> set[str]:
    """Every plain-name form a `class` line might use instead of a node's id (lower-cased):
    the label's first line, with and without a trailing "` vN`". A real generation has used
    both forms in place of the real id."""
    label = raw_bracket_content.strip()
    if label.startswith('"`') and label.endswith('`"'):
        label = label[2:-2]
    elif label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    first_line = label.split("\n", 1)[0].strip().lower()
    without_version = re.sub(r"\s+v\d+$", "", first_line).strip()
    return {first_line, without_version}


def fix_diagram_class_references(document: str) -> str:
    """Rebuilds §3's `classDef`/`class` lines from §4's own table, instead of repairing
    whatever the model wrote. §4 follows the **COMPONENT INCLUSION RULE** and is far more
    reliable than the diagram's own styling lines. Three real failures confirm patching each
    broken shape does not scale: a node named by its label instead of its id (sometimes with
    its version, which also breaks the whole diagram's parse); a `class` line with no matching
    `classDef`; and a component §4 marks `MODIFIED` with no `class` line at all. Deriving the
    styling from §4 removes the model's own `class`/`classDef` output from the trust boundary
    entirely.

    Reads §4's table (`NEW`/`MODIFIED`/`REMOVED` rows only) into `{name: className}`, resolves
    each name to a §3 node id (`_node_name_keys`), strips all existing `classDef`/`class` lines,
    and emits fresh ones grouped by class in `nodeNew`/`nodeModified`/`nodeRemoved` order. A §4
    name with no matching node is skipped, never guessed. §2 is never touched."""
    start_match = markdown_heading_pattern("## 3. Target Architecture").search(document)
    if start_match is None:
        return document
    section_start = start_match.start()
    end_match = markdown_heading_pattern("## 4. Affected Components").search(document, start_match.end())
    section_end = end_match.start() if end_match is not None else len(document)
    section = document[section_start:section_end]

    fence_start = section.find("```mermaid")
    if fence_start == -1:
        return document
    body_start = fence_start + len("```mermaid\n")
    fence_close = section.find("```", body_start)
    if fence_close == -1:
        return document
    diagram = section[body_start:fence_close]

    components_section = _markdown_section(document, "## 4. Affected Components", "## 5. Affected Data Contracts")
    status_by_name = {
        name.strip().lower(): _STATUS_TO_CLASS[status]
        for name, status in _AFFECTED_COMPONENT_ROW.findall(components_section)
    }
    if not status_by_name:
        return document  # §4 confirms no real change — nothing to color

    name_to_id: dict[str, str] = {}
    for m in _NODE_DECLARATION.finditer(diagram):
        for key in _node_name_keys(m.group(2)):
            name_to_id[key] = m.group(1)

    ids_by_class: dict[str, list[str]] = {}
    for name, class_name in status_by_name.items():
        node_id = name_to_id.get(name)
        if node_id:
            ids_by_class.setdefault(class_name, []).append(node_id)
    if not ids_by_class:
        return document  # no §4 change resolved to a node in this diagram

    body_lines = [
        line for line in diagram.splitlines()
        if line.strip() and not line.strip().startswith(("classDef", "class "))
    ]
    style_lines = [f"    {_STANDARD_CLASSDEFS[c]}" for c in _CLASS_ORDER if c in ids_by_class]
    style_lines += [
        f"    class {','.join(sorted(set(ids_by_class[c]), key=_node_id_sort_key))} {c}"
        for c in _CLASS_ORDER if c in ids_by_class
    ]
    new_diagram = "\n".join([*body_lines, *style_lines])

    abs_body_start = section_start + body_start
    abs_fence_close = section_start + fence_close
    return document[:abs_body_start] + new_diagram + "\n" + document[abs_fence_close:]


def previous_target_architecture_diagram(previous_adr_content: str | None) -> str:
    """Pulls the mermaid fence out of a previous ADR's "## 3. Target Architecture" section,
    minus any color coding (`strip_diagram_colors`). Returns `""` if there is no previous ADR,
    or its Target Architecture had no diagram."""
    if not previous_adr_content:
        return ""
    section = _markdown_section(previous_adr_content, "## 3. Target Architecture", "## 4. Affected Components")
    if "```mermaid" not in section:
        return ""
    diagram = section.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    return strip_diagram_colors(diagram)


def own_previous_architecture_diagram(adr_content: str | None) -> str:
    """Pulls the mermaid fence out of an ADR's own "## 2. Previous Architecture" — for
    redrafting the SAME unapproved draft (the "regenerate with feedback" endpoint), not for
    starting a new change. Reading §3 instead would wrongly turn this draft's own unfinalized
    target into a fake "previous" state; §2 reproduces the prior architecture unchanged by the
    new feedback.

    Returns `""` on a first-time run, or if §2 had no diagram. Also runs
    `strip_diagram_colors`, as a safety net even though §2 should never carry a color."""
    if not adr_content:
        return ""
    section = _markdown_section(adr_content, "## 2. Previous Architecture", "## 3. Target Architecture")
    if "```mermaid" not in section:
        return ""
    diagram = section.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    return strip_diagram_colors(diagram)


_PLACEHOLDER_PATTERNS = (
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\bTODO\b"),
    re.compile(r"^\s*\|\s*\|\s*$", re.MULTILINE),  # an empty `| |` table cell row
    re.compile(r"^\s*-\s*\[\s\]", re.MULTILINE),  # an unchecked template checkbox
    re.compile(r"\[(Describe|Explain|Insert|Add|List|Fill in)\b", re.IGNORECASE),
)


def adr_has_placeholder_leak(document: str) -> bool:
    """Mechanical backstop for generator.jinja's NO PLACEHOLDERS section, for when the model
    doesn't follow it. `agents.graph.synthesize_document` uses this to trigger a same-
    temperature retry, the same pattern `architecture_questions.service` uses for its own
    failure modes.

    Narrow check: `TBD`, `TODO`, HTML comments, empty table cells, unchecked checkboxes,
    leftover bracketed instructions. Does not judge content quality — that is the Critic's
    job."""
    return any(pattern.search(document) for pattern in _PLACEHOLDER_PATTERNS)


def adr_drops_suggest_info_content(document: str, qa_pairs: list[dict]) -> bool:
    """Backstop for generator.jinja's `[SUGGEST INFO]` handling: a reviewer authorizes a
    suggested, `LLM SUGGESTION:`-prefixed answer, but the model sometimes drops it silently —
    confirmed flaky at temperature=0, so `synthesize_document` retries here too.

    Returns `True` only when a `qa_pairs` answer is `[SUGGEST INFO]` and the document has zero
    `LLM SUGGESTION:` occurrences — a coarse check for the total-drop failure, not per-answer
    matching."""
    if not any(pair.get("answer") == "[SUGGEST INFO]" for pair in qa_pairs):
        return False
    return "LLM SUGGESTION:" not in document


# synthesize_document sends this as a follow-up turn when adr_drops_suggest_info_content fires.
# Naming the specific gap converges in one retry; a blind resample (used for placeholder leaks)
# took 3-4 attempts here in testing.
SUGGEST_INFO_CORRECTION_MESSAGE = (
    "Your draft above answers one or more clarifications marked `[SUGGEST INFO]` but never "
    "actually states the suggestion anywhere — no `LLM SUGGESTION:` line appears in the "
    "document at all. Revise the document: for every `[SUGGEST INFO]`-answered clarification, "
    "add a concrete, plausible answer prefixed with `LLM SUGGESTION:`, in the section it "
    "belongs to — including a diagram edge and a data-contract entry in §3/§5/§6 when the gap "
    "is about how two components interact, not only a prose sentence. Keep everything else "
    "unchanged. Return the FULL corrected document."
)
