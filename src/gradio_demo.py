from __future__ import annotations

import logging
import os
from typing import Callable, List, Tuple

from src.llm import GeminiClient, HuggingFaceClient, LLMRouter
from src.agent import run_agent
from src.schemas import Citation

logger = logging.getLogger(__name__)


def format_response(
    answer: str,
    citations: List[Citation],
    refusal: bool = False,
    needs_clarification: bool = False,
    clarification_question: str = "",
) -> str:
    try:
        if needs_clarification:
            return clarification_question

        if refusal:
            return answer

        if len(citations) == 0:
            return answer

        lines = [f"[{i}] {c.doc} | {c.section}" + (f" | p.{c.page}" if c.page else "")
                 for i, c in enumerate(citations, start=1)]
        return answer + "\n\n---\nSources:\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"format_response failed: {e}")
        return "Error formatting response."


def setup() -> Tuple[LLMRouter, Callable]:
    gemini_key = os.environ["GEMINI_API_KEY"]
    hf_key = os.environ["HF_API_KEY"]

    gemini_client = GeminiClient(api_key=gemini_key)
    hf_client = HuggingFaceClient(api_key=hf_key)
    llm_router = LLMRouter(primary=gemini_client, fallback=hf_client)

    try:
        from src.retrieval import build_retrieval_fn

        retrieval_fn = build_retrieval_fn()
        logger.info("Using real retrieval_fn from src.retrieval")
    except ImportError:
        logger.info(
            "src.retrieval not found (rag-layer not yet merged) — "
            "using stub retrieval_fn returning empty results"
        )
        retrieval_fn = lambda query: []

    return llm_router, retrieval_fn


def chat_fn(
    message: str,
    history: list,
    llm_router: LLMRouter,
    retrieval_fn: Callable,
) -> str:
    try:
        answer = run_agent(message, llm_router, retrieval_fn)
        return format_response(
            answer=answer.answer,
            citations=answer.citations,
            refusal=answer.refusal,
            needs_clarification=False,
            clarification_question="",
        )
    except Exception as e:
        logger.error(f"chat_fn failed: {e}")
        return "System error — please retry."
