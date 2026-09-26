"""TÉCNICA: Verificación de citación / grounding.

Qué problema resuelve: cuando un LLM redacta una respuesta a partir de varios hechos
recuperados, nada le impide "citar" un hecho que en realidad nunca se le mostró, o citar mal la
versión/entidad exacta — el modelo puede sonar seguro sin que la cita sea cierta. Confiar en la
cita del modelo sin comprobarla es justo el fallo de fiabilidad ("faithfulness") que hace que una
respuesta de RAG parezca bien fundamentada sin estarlo de verdad.

Cómo funciona aquí: `answer_question`/`answer_evolution_question` (en
`agents/stages/gold/service.py`) piden al LLM una salida estructurada con la respuesta MÁS una
lista de citas (`entity_type`, `entity_id`, `version` — ver `ChatCitation` en
`agents/stages/gold/schemas.py`). `_verify_citations` comprueba, de forma determinista — sin
otra llamada a un LLM — que cada cita corresponda a una fila que de verdad estaba entre las
recuperadas (`rows`). Cualquier cita que no encaje se descarta antes de llegar al usuario. Es el
mismo espíritu que `agents.shared.mentions_grounded_in_source` (que comprueba menciones de
componentes contra la transcripción), aplicado aquí a las citas del chat.

Quién la usa: `answer_question` y `answer_evolution_question`. Ver `.tmp/tasks2.md` tarea 1 y
`.tmp/advanced_techniques.md` §7."""

from collections.abc import Sequence

from agents.stages.gold.schemas import ChatCitation
from db.models import GoldEvolution


def _verify_citations(citations: list[ChatCitation], rows: Sequence[GoldEvolution]) -> list[ChatCitation]:
    """Keeps only the citations that match a row actually in `rows` — an LLM-reported
    `(entity_type, entity_id, version)` that was never retrieved is dropped, not trusted. Same
    "check it mechanically, do not just trust the model" spirit as
    `agents.shared.mentions_grounded_in_source`, applied here to citations instead of mentioned
    names. This is what makes a citation evidence instead of an unverified claim: the model can
    still phrase the answer badly, but it cannot cite a fact it was never shown."""
    known = {(row.entity_type, row.entity_id, row.version) for row in rows}
    return [c for c in citations if (c.entity_type, c.entity_id, c.version) in known]
