from __future__ import annotations

import logging
import re
from typing import Dict, List

import src.prompts as prompts
from src.schemas import AgentState, Citation
from src.contracts import ChunkDict, get_chunk_doc, get_chunk_section
from src.llm import LLMRouter, JSONParseError

logger = logging.getLogger(__name__)

REFUSAL_STRING = (
    "I don't have enough information in the corpus to answer "
    "this question."
)

CITATION_PATTERN = re.compile(r"\[Source:\s*([^|]+?)\s*\|\s*([^\]]+?)\]")


def classify_query(state: AgentState, llm: LLMRouter) -> dict:
    prompt = prompts.build_classification_prompt(state["query"])
    try:
        result = llm.generate_json(prompt)
        return {"query_type": result.get("query_type", "simple")}
    except JSONParseError:
        logger.warning("classify_query: JSONParseError, defaulting to simple")
        return {"query_type": "simple"}


def decompose_query(state: AgentState, llm: LLMRouter) -> dict:
    prompt = prompts.build_decomposition_prompt(state["query"])
    try:
        result = llm.generate_json(prompt)
        sub_queries = result.get("sub_queries", [])
        if not sub_queries:
            return {"sub_queries": [state["query"]]}
        return {"sub_queries": sub_queries[:3]}
    except JSONParseError:
        logger.warning("decompose_query: JSONParseError, using original query")
        return {"sub_queries": [state["query"]]}


def handle_refusal(state: AgentState) -> dict:
    return {
        "refusal": True,
        "confidence": 0.0,
        "answer": REFUSAL_STRING,
    }


def handle_clarification(state: AgentState, llm: LLMRouter) -> dict:
    clarification_prompt = (
        f"The following query is ambiguous: '{state['query']}'. "
        f"Generate a single clarifying question to ask the user."
    )
    question = llm.generate(clarification_prompt, "")
    return {
        "needs_clarification": True,
        "clarification_question": question,
    }


def synthesize_answer(state: AgentState, llm: LLMRouter) -> dict:
    chunks = state["retrieved_chunks"]
    system_prompt = prompts.build_system_prompt(chunks)
    answer_text = llm.generate(system_prompt, state["query"])

    citations = []
    for match in CITATION_PATTERN.finditer(answer_text):
        doc, section = match.group(1).strip(), match.group(2).strip()
        citations.append(Citation(doc=doc, section=section, snippet=""))

    confidence = max((c["score"] for c in chunks), default=0.0)

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence": confidence,
    }


def validate_citations(state: AgentState) -> dict:
    if state["answer"].strip() == REFUSAL_STRING:
        return {"validation_passed": True}

    if state["retry_count"] >= 1:
        return {"validation_passed": True}

    chunk_text_combined = " ".join(
        c["text"].lower() for c in state["retrieved_chunks"]
    )
    chunk_tokens = set(chunk_text_combined.split())

    answer_tokens = set(state["answer"].lower().split())
    if not answer_tokens:
        overlap = 0.0
    else:
        overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)

    if overlap >= 0.25:
        return {"validation_passed": True}
    else:
        return {
            "validation_passed": False,
            "retry_count": state["retry_count"] + 1,
        }


def route_by_query_type(state: AgentState) -> str:
    return state["query_type"]


def route_by_validation(state: AgentState) -> str:
    if state["validation_passed"]:
        return "passed"
    if state["retry_count"] >= 1:
        return "passed"
    return "retry"
