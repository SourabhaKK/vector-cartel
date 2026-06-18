from typing import List, Optional, TypedDict

from pydantic import BaseModel, computed_field, field_validator

from src.contracts import ChunkDict, InputScanResult


class Citation(BaseModel):
    doc: str
    section: str
    page: Optional[int] = None
    snippet: str

    @field_validator("snippet")
    @classmethod
    def snippet_max_200_chars(cls, value: str) -> str:
        if len(value) > 200:
            raise ValueError("snippet must not exceed 200 characters")
        return value


class SecureOpsAnswer(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    refusal: bool = False
    query_type: str = "simple"

    @field_validator("confidence")
    @classmethod
    def confidence_within_bounds(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return value

    @computed_field
    @property
    def sources_used(self) -> List[str]:
        seen = []
        for citation in self.citations:
            if citation.doc not in seen:
                seen.append(citation.doc)
        return seen


class AgentState(TypedDict):
    query: str
    query_type: str
    sub_queries: List[str]
    retrieved_chunks: List[ChunkDict]
    answer: str
    citations: List[Citation]
    confidence: float
    refusal: bool
    needs_clarification: bool
    clarification_question: str
    retry_count: int
    error: Optional[str]
    validation_passed: bool
