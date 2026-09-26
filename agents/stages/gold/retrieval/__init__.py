"""Cada técnica de recuperación (RAG) que usa el chat de Gold vive en su propio fichero aquí,
una por técnica, para poder estudiarlas y modificarlas de forma aislada:

- `hybrid_search.py` — búsqueda vectorial + léxica, fusionadas con Reciprocal Rank Fusion.
- `deduplication.py` — colapsar varias versiones de la misma entidad a una sola en el top-k.
- `reranking.py` — repuntuar candidatos con un cross-encoder local.
- `citation_verification.py` — comprobar que cada cita del LLM corresponde a una fila real.
- `conversational_memory.py` — la mitad de generación de la memoria conversacional del chat.
- `corrective_rag.py` — reintento automático con un filtro de relevancia más laxo.
- `query_expansion.py` — reformular la pregunta para encontrar sinónimos sin palabras en común.

`agents/stages/gold/service.py` es quien orquesta la mayoría (`top_k_gold_evolution`,
`answer_question`, `answer_evolution_question`); `corrective_rag.py` es la excepción — la
orquesta directamente el endpoint `/chat` en `app/routers/frontend.py`, envolviendo su llamada
a `top_k_gold_evolution`. Este paquete no decide CUÁNDO se usa cada técnica, solo implementa
CÓMO funciona cada una."""
