"""This file holds callable helpers for the ADR-generation stage. These are not the prompt
builder itself. They do three things: pull a previous ADR's own diagrams out for grounding,
strip display styling out of a diagram before it gets reused, and act as a mechanical backstop
for the prompt's own "no placeholders" rule."""

import re

from agents.shared import _markdown_section


def strip_diagram_colors(diagram: str) -> str:
    """Drops every `classDef` and `class` styling line from a Mermaid diagram string. A color
    class marks what changed in a diagram, or shows how Gold renders it. See
    `prompts/adr_generation/generator.jinja`'s DIAGRAM COLOR CODING section, and
    `agents.stages.gold.service.current_architecture_diagram`'s own `goldNode` styling. A color
    class must never survive into a later document as if it were still true.

    `previous_target_architecture_diagram` and `own_previous_architecture_diagram` (both below)
    use this function. `agents.graph.synthesize_document` also uses it on Gold's own diagram.
    Each of these feeds an ADR's "## 2. Previous Architecture" section. That section must always
    be a plain, colorless snapshot. It must never carry styling from the diagram's own source.

    This function is public, with no leading underscore, because `agents.graph` uses it too, not
    just this module."""
    lines = [line for line in diagram.splitlines() if not line.strip().startswith(("classDef", "class "))]
    return "\n".join(lines).strip()


def previous_target_architecture_diagram(previous_adr_content: str | None) -> str:
    """Pulls the ` ```mermaid ` fence out of a previous ADR's own "## 3. Target Architecture"
    section. This run's "Previous Architecture" is exactly what the last run's "Target
    Architecture" already was, minus any color coding. See `strip_diagram_colors` for that part.
    Returns `""` if there is no previous ADR. Also returns `""` if the Target Architecture
    section had no diagram. That second case happens on a first-time run, or when nothing about
    the architecture was ever confirmed."""
    if not previous_adr_content:
        return ""
    section = _markdown_section(previous_adr_content, "## 3. Target Architecture", "## 4. Affected Components")
    if "```mermaid" not in section:
        return ""
    diagram = section.split("```mermaid", 1)[1].split("```", 1)[0].strip()
    return strip_diagram_colors(diagram)


def own_previous_architecture_diagram(adr_content: str | None) -> str:
    """Pulls the ` ```mermaid ` fence out of an ADR's **own** "## 2. Previous Architecture"
    section. This is for redrafting the SAME still-unapproved draft. The frontend's "regenerate
    with feedback" endpoint uses it for that. It is not for starting a new change from an
    already-approved one.

    That endpoint's `previous` value is this very draft's latest `SilverDocument` row. That row
    was already saved by `write_document` before any human reviewed it. Reading its §3 Target
    Architecture instead (via `previous_target_architecture_diagram`) would wrongly turn this
    unapproved draft's own target into a fake "previous" state, for a change nothing has
    finalized yet. Reading its §2 instead reproduces exactly what this draft already set as the
    prior architecture, unchanged by the extra feedback.

    Returns `""` if there is no ADR (a first-time run). Also returns `""` if its §2 had no
    diagram, meaning nothing about the prior architecture is established either.

    This also runs `strip_diagram_colors`. Section §2 should never carry a color class in the
    first place. But this gives the same extra safety guarantee that
    `previous_target_architecture_diagram` gives its own output, in case an earlier generation
    let one slip in anyway."""
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
    """This is a mechanical backstop for `prompts/adr_generation/generator.jinja`'s own NO
    PLACEHOLDERS section. That section only works if the model follows the prompt. Nothing
    downstream catches a template artifact that slips through anyway, until now.

    `agents.graph.synthesize_document` uses this function to trigger a retry at the same
    temperature. This is the same pattern `agents.stages.architecture_questions.service`'s
    `_looks_shallow` and `_ungrounded_component_names` already use for their own stage's failure
    mode. This applies that same discipline to ADR generation. Before this, ADR generation's only
    quality gate was the `agents.stages.adr_generation.testing` golden set, which uses a real LLM and does not run in
    CI.

    This check is deliberately narrow. It catches mechanical leaks: `TBD`, `TODO`, HTML comments,
    empty table cells, unchecked checkboxes, and leftover bracketed instructions. These are the
    kind of leaks a template-following slip produces. This check does not judge whether the
    document's CONTENT is actually good. That judgment is still the Critic's job, not this
    function's."""
    return any(pattern.search(document) for pattern in _PLACEHOLDER_PATTERNS)
