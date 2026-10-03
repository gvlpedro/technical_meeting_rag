"""This file holds callable helpers for the ADR-generation stage. These are not the prompt
builder itself. They do four things: pull a previous ADR's own diagrams out for grounding, strip
display styling out of a diagram before it gets reused, deterministically rebuild §3's node
coloring from §4's own table instead of trusting the model's own `class`/`classDef` lines, and
act as a mechanical backstop for the prompt's own "no placeholders" rule."""

import re

from agents.shared import _markdown_section, markdown_heading_pattern


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


# Matches a node declaration under either label style this stage's own output actually uses:
# Gold's backtick "markdown string" style (`n0["`Frontend v1\n*...*`"]`, reproduced verbatim from
# `PREVIOUS_ARCHITECTURE_DIAGRAM`) or plain Mermaid brackets (`CheckoutService[Checkout Service]`,
# the model's own style when there is no previous diagram to reproduce). `.*?` + `DOTALL` lets the
# bracket content span a literal newline either way; `_node_name_keys` unwraps whichever quoting
# this particular node happened to use.
_NODE_DECLARATION = re.compile(r"(\w+)\[(.*?)\]", re.DOTALL)
# One row of §4's own table: `| Component | **STATUS** | Description |`. Deliberately requires
# the bold `**...**` status marker the prompt's own table format always uses, so this can never
# mistake some unrelated `|`-containing line (a different table entirely) for a component row.
_AFFECTED_COMPONENT_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*\*\*(NEW|MODIFIED|REMOVED)\*\*\s*\|", re.MULTILINE)
_STATUS_TO_CLASS = {"NEW": "nodeNew", "MODIFIED": "nodeModified", "REMOVED": "nodeRemoved"}
# This fixed order is the order `prompts/adr_generation/generator.jinja`'s own example declares
# classes in — kept here too so a rebuilt diagram's `classDef`/`class` lines come out in the same
# order a correct generation would have used, not because Mermaid itself cares.
_CLASS_ORDER = ("nodeNew", "nodeModified", "nodeRemoved")
# The fixed palette the prompt's DIAGRAM COLOR CODING section itself specifies, verbatim — the
# only three `classDef`s a diagram from this stage ever needs.
_STANDARD_CLASSDEFS = {
    "nodeNew": "classDef nodeNew fill:#34d399,stroke:#047857,color:#022c22",
    "nodeModified": "classDef nodeModified fill:#fb923c,stroke:#c2410c,color:#431407",
    "nodeRemoved": "classDef nodeRemoved fill:#f87171,stroke:#b91c1c,color:#450a0a,stroke-dasharray: 5 5",
}


def _node_id_sort_key(node_id: str) -> tuple[int, str]:
    """Sorts `n0` before `n1` before `n10` numerically when the id ends in digits (the only
    style this stage's own node ids ever use), falling back to plain alphabetical order for
    anything else — never a crash either way."""
    match = re.search(r"\d+$", node_id)
    return (int(match.group()), node_id) if match else (-1, node_id)


def _node_name_keys(raw_bracket_content: str) -> set[str]:
    """Every plain-name form a `class` line might use instead of a node's real id, lower-cased
    for a case-insensitive match. Strips the backtick-markdown-string quoting if present
    (`` "`...`" ``) or plain double quotes, and takes the first line only (drops the italic
    ADR-ref subtitle a backtick label can carry) — but returns BOTH that first line as-is AND
    with a trailing "` vN`" stripped, since a real generation has been seen using either form
    (`class ia-service nodeNew` — no version at all — and `class backend v1 nodeModified` — the
    node's own full label, version included, in place of its id)."""
    label = raw_bracket_content.strip()
    if label.startswith('"`') and label.endswith('`"'):
        label = label[2:-2]
    elif label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    first_line = label.split("\n", 1)[0].strip().lower()
    without_version = re.sub(r"\s+v\d+$", "", first_line).strip()
    return {first_line, without_version}


def fix_diagram_class_references(document: str) -> str:
    """Rebuilds §3 Target Architecture's `classDef`/`class` lines from scratch, from §4 Affected
    Components' own table, instead of trying to repair whatever the model wrote for them. §4 is
    governed by the **COMPONENT INCLUSION RULE** and is already far more reliable than the
    diagram's own styling lines — three real, reported failures confirm that reliably coloring a
    diagram by trying to patch the model's own `class`/`classDef` output is a losing game, each
    one a different way the SAME underlying intent ("Backend is `MODIFIED`, color it") failed to
    reach the diagram at all:

    1. `class ia-service nodeNew` / `class frontend nodeModified` / `class backend v1
       nodeModified` — the model named a node by its own label (with or without its version)
       instead of the id it had itself just declared. Mermaid's `class` directive only resolves a
       declared node id, never a label string: the first two left the node uncolored; the third
       broke the WHOLE diagram's parse (no `SPACE` token is valid inside a bare identifier list).
    2. `class n1 nodeModified` with no `classDef nodeModified ...` declared anywhere in the same
       diagram — nothing for Mermaid to apply.
    3. §4 correctly says `Frontend` is `**MODIFIED**`, but §3 has no `class` line for its node at
       all — the model simply never wrote one, despite the fact being right there in §4.

    Patching each shape individually, as 1 and 2 above already were, only ever covers what has
    already been seen go wrong, and a 4th, 5th shape of the same underlying failure is exactly as
    likely as the 3rd was. Deriving the whole thing from §4 instead removes the model's own
    `class`/`classDef` output from the trust boundary entirely — there is no longer a shape of
    mistake IN those lines left to patch, because this never reads them.

    Reads §4's table once (`_AFFECTED_COMPONENT_ROW`): `{component name (lower-cased): className}`
    for every `NEW`/`MODIFIED`/`REMOVED` row (never `UNCHANGED` — §4 no longer lists those at
    all). Reads §3's own node declarations once (`_NODE_DECLARATION`) into the same name-matching
    map `fix_diagram_class_references`'s own node-id resolution already used
    (`_node_name_keys`), to resolve each §4 name to its real node id. Then strips every existing
    `classDef`/`class` line from §3's diagram (ignoring whatever the model wrote there) and
    emits fresh ones: one `classDef` per class actually needed, one `class` line per class
    grouping all of its node ids together (same convention the prompt's own example uses), in the
    fixed `nodeNew`/`nodeModified`/`nodeRemoved` order. A §4 name that cannot be resolved to any
    node in §3 is silently skipped — this only colors a node it can unambiguously identify, never
    guesses. §2 Previous Architecture is never touched — this only ever looks inside §3."""
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
        return document  # §4 confirms no real change at all — nothing this diagram needs colored

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
        return document  # none of §4's confirmed changes resolved to a node in this diagram

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


def adr_drops_suggest_info_content(document: str, qa_pairs: list[dict]) -> bool:
    """This is a mechanical backstop for `prompts/adr_generation/generator.jinja`'s own
    `[SUGGEST INFO]` handling — the same kind of backstop `adr_has_placeholder_leak` already is
    for the NO PLACEHOLDERS section, applied to a second, real, reported failure mode: a
    reviewer answers a clarification `[SUGGEST INFO]` (explicitly authorizing a suggested,
    `LLM SUGGESTION:`-prefixed answer for that one gap), but the model sometimes drops it
    silently instead — no suggestion anywhere in the document, no diagram edge, nothing. This
    was confirmed to actually happen at temperature=0, and to be flaky: the exact same
    transcript/clarifications produced a correct suggestion (and a diagram edge) on one run and
    dropped it entirely on the next two, with zero code changes between runs. This is exactly
    why `agents.graph.synthesize_document` needs a retry trigger here, the same as it already
    has for a placeholder leak — temperature=0 makes a bad run just as reproducible as a good
    one.

    Returns `True` (retry-worthy) only when at least one `qa_pairs` answer is the literal
    `[SUGGEST INFO]` marker AND the document contains zero `LLM SUGGESTION:` occurrences at all.
    This is deliberately a coarse, binary check — it does not try to match each `[SUGGEST INFO]`
    answer to its own specific suggestion, the same way `adr_has_placeholder_leak` does not
    judge content quality. It only catches the total-silent-drop failure actually observed."""
    if not any(pair.get("answer") == "[SUGGEST INFO]" for pair in qa_pairs):
        return False
    return "LLM SUGGESTION:" not in document


# `agents.graph.synthesize_document` sends this as a follow-up user turn, after the model's own
# dropped draft, when `adr_drops_suggest_info_content` fires. A blind resample at a higher
# temperature (the same recovery `adr_has_placeholder_leak` uses) measurably works here too, but
# real testing showed it needs 3-4 attempts on a genuinely hard case — leaving a real, if small,
# chance of exhausting all `SHALLOW_RETRY_ATTEMPTS` before recovering, which is exactly what a
# real user hit. Naming the specific, actual problem (as a human reviewer would) instead of
# re-rolling blind measurably converges in one retry instead of three or four, in the same real
# testing. This is deliberately never used for a plain placeholder leak — that failure mode has
# no specific gap to name, so a blind resample stays the right (and already-proven) fix there.
SUGGEST_INFO_CORRECTION_MESSAGE = (
    "Your draft above answers one or more clarifications marked `[SUGGEST INFO]` but never "
    "actually states the suggestion anywhere — no `LLM SUGGESTION:` line appears in the "
    "document at all. Revise the document: for every `[SUGGEST INFO]`-answered clarification, "
    "add a concrete, plausible answer prefixed with `LLM SUGGESTION:`, in the section it "
    "belongs to — including a diagram edge and a data-contract entry in §3/§5/§6 when the gap "
    "is about how two components interact, not only a prose sentence. Keep everything else "
    "unchanged. Return the FULL corrected document."
)
