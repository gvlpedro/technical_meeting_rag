"""TÉCNICA: Memoria conversacional.

Qué problema resuelve: sin esto, cada pregunta del chat se trata como si fuera la primera — una
pregunta de seguimiento tipo "¿y quién lo aprobó?" no tiene forma de saber a qué se refiere "lo",
porque no queda ningún rastro del turno anterior de la conversación.

Cómo funciona aquí, en dos mitades que viven en dos ficheros distintos porque sirven a dos pasos
diferentes del pipeline:

1. **Mitad de recuperación** — `_contextualize_question`, en `app/routers/frontend.py` (no en
   este fichero: necesita el tipo `ChatMessage` de ese router, así que se queda en esa capa).
   Antes de embeber/buscar, prefija la pregunta con los últimos turnos de la conversación, SOLO
   para que la búsqueda (por embedding y por texto) encuentre las filas correctas — nunca para
   que el LLM conteste con eso.
2. **Mitad de generación** — `_format_history_block`, aquí mismo. Vuelve a mostrar esos mismos
   últimos turnos al LLM, pero ya en la llamada que redacta la respuesta final, con una
   instrucción explícita en el prompt: el historial SOLO puede usarse para resolver a qué se
   refiere la pregunta (un pronombre, "eso", "ese componente") — nunca como fuente de hechos.
   Cada hecho de la respuesta debe seguir viniendo de las filas recuperadas, no del historial.

Quién la usa: `answer_question` y `answer_evolution_question`. Ver
`.tmp/advanced_techniques.md` §8."""

# `answer_question`/`answer_evolution_question` only ever render this many of the most recent
# `history` messages, even if a caller passes more. This bounds prompt size and cost for a
# long-running chat session — resolving a follow-up question only ever needs a few turns of
# context, never the whole conversation since login.
MAX_HISTORY_MESSAGES = 6


def _format_history_block(history: list[tuple[str, str]] | None) -> str:
    """Renders the last `MAX_HISTORY_MESSAGES` of `history` — a list of `(role, content)` pairs,
    `role` one of `"user"`/`"assistant"`, oldest first, matching `frontend/app.py`'s own
    `chat_history` shape — as one labeled block for a prompt. Returns `""` when there is no
    history, so a caller can always concatenate this in without an `if` of its own."""
    if not history:
        return ""
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    lines = "\n".join(f"{role.capitalize()}: {content}" for role, content in trimmed)
    return f"Previous conversation (most recent last):\n{lines}\n\n"
