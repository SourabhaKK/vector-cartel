"""
gradio_demo.py — Demo interface layer for SecureOps Assistant.

This module contains the business logic underneath the Gradio
UI: formatting responses, initialising clients, and wrapping
run_agent with error handling. The actual gr.ChatInterface
construction and .launch() call live in notebooks/, NOT here —
that keeps this module unit-testable with pytest, since
gr.ChatInterface requires a running server/browser to test.

THREE FUNCTIONS:
  format_response() — turns a SecureOpsAnswer's fields into
                       display-ready markdown text
  setup()            — reads API keys from env, constructs
                       LLMRouter, locates retrieval_fn (falls
                       back to a stub if rag-layer not yet merged)
  chat_fn()           — the actual function gr.ChatInterface calls
                       per message; wraps run_agent + format_response
                       with broad exception handling so a crash
                       inside the agent never takes down the UI

FALSY VS NONE NOTE:
format_response checks len(citations) == 0, not "if not citations".
The latter treats None and [] identically, which silently masked
a bug where citations=None should raise rather than render as if
there were simply no sources. len() forces an explicit TypeError
on None, which the broad except Exception then correctly converts
into the "error" string tests assert on.

MODULE-QUALIFIED IMPORT NOTE:
Unlike src/agent.py's "import src.prompts as prompts" pattern,
this file uses direct imports (from src.llm import GeminiClient).
That earlier fix was needed because agent.py tests spy on a
FUNCTION being called mid-execution (mocker.patch intercepting
prompts.build_x's call). Here, mocker.patch REPLACES CLASSES
wholesale (GeminiClient, HuggingFaceClient) — any reference to
the class name resolves to the patched mock automatically,
regardless of import style, because the patch target is the
name binding itself, not a call-time lookup. Do not apply the
module-qualified pattern here — it solves a different problem
than the one present in this file.
"""

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
    """
    Turns a SecureOpsAnswer's fields into display-ready markdown.

    Precedence: clarification question (if needs_clarification),
    then plain answer (if refusal or no citations), then answer
    plus a numbered "Sources:" block (if citations is non-empty).

    Args:
        answer: The generated answer text.
        citations: List of Citation objects, or None/malformed —
                  any failure to process these is caught below.
        refusal: True if the agent refused to answer.
        needs_clarification: True if the agent needs more info.
        clarification_question: The question to show the user
                                when needs_clarification is True.

    Returns:
        Markdown string for display, or "Error formatting
        response." if anything above raised (e.g. citations=None).

    Called by: chat_fn, after run_agent returns a SecureOpsAnswer.
    """
    try:
        if needs_clarification:
            return clarification_question

        if refusal:
            return answer

        if len(citations) == 0:
            return answer

        return answer + "\n\n---\nSources:\n" + _format_citation_list(citations)
    except Exception as e:
        logger.error(f"format_response failed: {e}")
        return "Error formatting response."


def _format_citation_list(citations: List[Citation]) -> str:
    """
    Formats a list of Citation objects into a numbered markdown
    list for display under the "Sources:" header.

    Each line: "[{index}] {doc} | {section}" with page appended
    in parentheses if present.

    Args:
        citations: Non-empty list of Citation objects. Caller
                   (format_response) is responsible for checking
                   emptiness before calling this — this helper
                   assumes at least one citation exists.

    Returns:
        Multi-line string, one citation per line, 1-indexed.
    """
    lines = []
    for i, citation in enumerate(citations, start=1):
        page_part = f" (p.{citation.page})" if citation.page else ""
        lines.append(f"[{i}] {citation.doc} | {citation.section}{page_part}")
    return "\n".join(lines)


def _get_required_env(key: str) -> str:
    """
    Reads a required environment variable, raising a clear
    error if missing rather than silently passing None to
    GeminiClient/HuggingFaceClient and failing with a confusing
    downstream error during the live demo.
    """
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Set it in Colab Secrets before calling setup()."
        )
    return value


def setup() -> Tuple[LLMRouter, Callable]:
    """
    Initialises the LLMRouter and retrieval_fn for a demo session.

    Reads GEMINI_API_KEY and HF_API_KEY from the environment (raises
    EnvironmentError via _get_required_env if either is missing),
    constructs GeminiClient as primary and HuggingFaceClient as
    fallback, and wraps them in an LLMRouter.

    Attempts to import build_retrieval_fn from src.retrieval
    (rag-layer's output). Falls back to a stub returning [] if
    rag-layer hasn't merged into dev yet — this lets the demo run
    end-to-end (with empty retrieval) before all branches merge.

    Returns:
        (llm_router, retrieval_fn) tuple, both passed into chat_fn.

    Raises:
        EnvironmentError: If GEMINI_API_KEY or HF_API_KEY is unset.

    Called by: notebooks/secureops_pipeline.ipynb, once at startup.
    """
    gemini_key = _get_required_env("GEMINI_API_KEY")
    hf_key = _get_required_env("HF_API_KEY")

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
    """
    The function gr.ChatInterface invokes once per user message.

    Wraps run_agent (the full agent pipeline) with broad exception
    handling — any failure anywhere in the agent graph (LLM timeout,
    retrieval failure, malformed state) is caught here and converted
    to a user-facing error string rather than crashing the Gradio
    server mid-demo.

    Args:
        message: The user's typed query.
        history: Gradio's conversation history (unused currently —
                run_agent treats each query independently, no
                multi-turn context yet).
        llm_router: Injected LLMRouter from setup().
        retrieval_fn: Injected retrieval function from setup().

    Returns:
        Markdown-formatted answer string, or an error string if
        anything in the pipeline raised.

    Called by: notebooks/secureops_pipeline.ipynb, via
        gr.ChatInterface(lambda m, h: chat_fn(m, h, llm_router, retrieval_fn))
    """
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
