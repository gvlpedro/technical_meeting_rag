"""TÉCNICA: Expansión de consulta (Query Expansion).

Qué problema resuelve: la búsqueda híbrida (vectorial + léxica, ver `hybrid_search.py`) ya
rescata el caso de un nombre propio o acrónimo que el embedding difumina — la búsqueda léxica lo
encuentra igual. Pero hay un caso que ni la vectorial ni la léxica resuelven solas: cuando el
usuario nombra una entidad con un sinónimo que no comparte ni una palabra con el nombre canónico
— "el módulo de pagos" frente al nombre real "Payments Gateway". Ahí la búsqueda léxica no
encuentra nada (cero palabras en común) y la vectorial depende de lo bien que el embedding capte
esa sinonimia, que no siempre es fiable.

Cómo funciona aquí: `expand_question` hace una llamada barata al LLM, con temperatura baja pero
no cero (a propósito — ver su propio docstring), pidiendo 2-3 reformulaciones de la misma
pregunta, nunca una pregunta distinta. Cada reformulación se embebe y se busca igual que la
pregunta original, y todos esos rankings se fusionan con el mismo mecanismo de Reciprocal Rank
Fusion que ya usa la búsqueda híbrida (`_reciprocal_rank_fusion`) — antes de deduplicar y
reordenar, no después: una reformulación que encuentra la entidad correcta debe poder competir
por un hueco en el top-k igual que cualquier otra señal.

Quién la orquesta: `top_k_gold_evolution` (en `agents/stages/gold/service.py`) la activa con
`expand=True` (opt-in, apagado por defecto — es una llamada a LLM extra por pregunta). Ver
`.tmp/tasks2.md` tarea 3 y `.tmp/advanced_techniques.md` §4."""

from agents.stages.gold.schemas import QuestionExpansion
from agents.template import load_json_response
from llm import router

# How many reformulations to ask for. 2-3 is enough to catch a genuine synonym without paying
# for (and fusing in the noise of) a large batch of near-duplicate rewordings.
QUESTION_EXPANSION_COUNT = 3


async def expand_question(question: str) -> list[str]:
    """Asks a cheap LLM call for `QUESTION_EXPANSION_COUNT` alternative phrasings of
    `question` — synonyms and reformulations, never a change of what is actually being asked.
    Returns only the NEW reformulations, never the original question itself — the caller
    already has that and decides how to combine them (see `top_k_gold_evolution`'s
    `expand=True`).

    Temperature is low but deliberately not 0: this call's whole job is to produce wording
    DIFFERENT from the original question, and a temperature=0 call tends to paraphrase only
    superficially (word order, a synonym or two) instead of genuinely varying the phrasing —
    the opposite of what widening retrieval needs. This is the inverse of why other calls in
    this codebase (`classify_questions`, `answer_question`) use `temperature=0`: those need the
    SAME answer every time; this one needs a few genuinely DIFFERENT ones."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Write {QUESTION_EXPANSION_COUNT} alternative phrasings of the question below, "
                "in the same language as the question. Each one must ask for exactly the same "
                "information as the original — never a different or broader question, only "
                "reworded: swap a term for a plausible synonym or a more/less formal name for "
                "the same thing. Do not include the original question itself in the list.\n\n"
                f"Question: {question}"
            ),
        }
    ]
    response = await router.complete(messages, response_format=QuestionExpansion, temperature=0.3)
    result = QuestionExpansion.model_validate(load_json_response(response.choices[0].message.content))
    return result.reformulations
