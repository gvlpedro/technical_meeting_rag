"""TECHNIQUE: Conversation memory.

Problem: without this, each chat question stands alone. A follow-up like "who approved it?"
has no way to know what "it" means.

Two halves, in two files, for two different pipeline steps:

1. **Retrieval half** — `_contextualize_question` in `app/routers/frontend.py`. Prefixes the
   question with recent turns before search, only to help retrieval find the right rows.
2. **Generation half** — `_format_history_block`, here. Shows the same recent turns to the LLM
   that drafts the final answer. The prompt says: use history only to resolve what the
   question refers to, never as a source of facts. Every fact must still come from the
   retrieved rows.

Used by: `answer_question`, `answer_evolution_question`."""

# Caps how many recent `history` messages this renders. Bounds prompt size and cost. A
# follow-up question only needs a few turns of context.
MAX_HISTORY_MESSAGES = 6


def _format_history_block(history: list[tuple[str, str]] | None) -> str:
    """Renders the last `MAX_HISTORY_MESSAGES` of `history` as one labeled block for a prompt.
    `history` is a list of `(role, content)` pairs, oldest first. Returns `""` when there is no
    history."""
    if not history:
        return ""
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    lines = "\n".join(f"{role.capitalize()}: {content}" for role, content in trimmed)
    return f"Previous conversation (most recent last):\n{lines}\n\n"
