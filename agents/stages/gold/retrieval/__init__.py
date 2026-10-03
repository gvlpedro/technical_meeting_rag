"""Each retrieval technique for the Gold chat lives in its own file here:

- `hybrid_search.py` — vector + lexical search, fused with Reciprocal Rank Fusion.
- `deduplication.py` — collapses multiple versions of one entity into the top-k.
- `reranking.py` — re-scores candidates with a local cross-encoder.
- `citation_verification.py` — checks each LLM citation against a real row.
- `conversational_memory.py` — builds the chat's conversation memory.
- `corrective_rag.py` — retries with a looser relevance filter.
- `query_expansion.py` — rephrases the question to find synonyms.

`agents/stages/gold/service.py` calls most of these. `app/routers/frontend.py`'s `/chat`
endpoint calls `corrective_rag.py` directly. This package defines HOW each technique works,
not WHEN to use it."""
