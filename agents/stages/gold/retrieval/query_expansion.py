"""TECHNIQUE: Query expansion.

Problem: hybrid search still misses a synonym that shares no word with the canonical name
("the payments module" vs. "Payments Gateway"). Lexical search finds nothing; vector search
depends on the embedding capturing that synonym, which is not always reliable.

How: `expand_question` asks the LLM for 2-3 alternative phrasings of the same question, never
a different question. Each reformulation is embedded and searched like the original, and all
rankings fuse through the same Reciprocal Rank Fusion hybrid search uses.

Used by: `top_k_gold_evolution`, with `expand=True` (opt-in, off by default — one extra LLM
call per question)."""

from agents.stages.gold.schemas import QuestionExpansion
from agents.template import load_json_response
from llm import router

# How many reformulations to ask for. 2-3 catches a real synonym without adding noise.
QUESTION_EXPANSION_COUNT = 3


async def expand_question(question: str) -> list[str]:
    """Asks the LLM for `QUESTION_EXPANSION_COUNT` alternative phrasings of `question`.
    Returns only the new reformulations, never the original question.

    Temperature is low but not 0: a temperature=0 call tends to paraphrase only superficially.
    This call needs genuinely different wording, unlike `classify_questions`/`answer_question`,
    which need the same answer every time."""
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
