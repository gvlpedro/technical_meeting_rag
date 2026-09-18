"""This module re-exports the Gold retrieval functions from agents.stages.gold.service.

This suite uses these functions to search Gold by similarity.
scripts/chat_gold.py uses the same functions.
This keeps one single implementation of Gold retrieval, so test code and
production code cannot drift apart.
"""

from agents.stages.gold.service import answer_question, embed_question, top_k_gold_evolution

__all__ = ["answer_question", "embed_question", "top_k_gold_evolution"]
