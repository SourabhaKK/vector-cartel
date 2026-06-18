import pytest
from pydantic import ValidationError


def test_citation_valid_construction():
    from src.schemas import Citation

    citation = Citation(
        doc="NIST SP 800-82 Rev 3",
        section="5.2.3",
        page=42,
        snippet="firewall controls text",
    )

    assert citation.doc == "NIST SP 800-82 Rev 3"
    assert citation.section == "5.2.3"
    assert citation.page == 42
    assert citation.snippet == "firewall controls text"


def test_citation_snippet_max_200_chars():
    from src.schemas import Citation

    with pytest.raises(ValidationError):
        Citation(
            doc="NIST SP 800-82 Rev 3",
            section="5.2.3",
            page=42,
            snippet="x" * 201,
        )


def test_citation_page_optional():
    from src.schemas import Citation

    citation = Citation(
        doc="NIST SP 800-82 Rev 3",
        section="5.2.3",
        snippet="firewall controls text",
    )

    assert citation.page is None


def test_secure_ops_answer_valid_construction():
    from src.schemas import Citation, SecureOpsAnswer

    answer = SecureOpsAnswer(
        answer="Test answer",
        citations=[
            Citation(
                doc="NIST SP 800-82 Rev 3",
                section="5.2.3",
                snippet="test snippet",
            )
        ],
        confidence=0.85,
        refusal=False,
        query_type="simple",
    )

    assert answer.answer == "Test answer"
    assert len(answer.citations) == 1
    assert answer.confidence == 0.85
    assert answer.refusal is False
    assert answer.query_type == "simple"


def test_secure_ops_answer_sources_deduplicates():
    from src.schemas import Citation, SecureOpsAnswer

    answer = SecureOpsAnswer(
        answer="Test answer",
        citations=[
            Citation(
                doc="NIST SP 800-82 Rev 3",
                section="5.2.3",
                snippet="test snippet one",
            ),
            Citation(
                doc="NIST SP 800-82 Rev 3",
                section="5.2.4",
                snippet="test snippet two",
            ),
        ],
        confidence=0.85,
        refusal=False,
        query_type="simple",
    )

    assert answer.sources_used == ["NIST SP 800-82 Rev 3"]


def test_confidence_below_zero_raises():
    from src.schemas import SecureOpsAnswer

    with pytest.raises(ValidationError):
        SecureOpsAnswer(
            answer="Test answer",
            citations=[],
            confidence=-0.1,
            refusal=False,
            query_type="simple",
        )


def test_confidence_above_one_raises():
    from src.schemas import SecureOpsAnswer

    with pytest.raises(ValidationError):
        SecureOpsAnswer(
            answer="Test answer",
            citations=[],
            confidence=1.1,
            refusal=False,
            query_type="simple",
        )


def test_agent_state_default_values():
    from src.schemas import AgentState

    state: AgentState = {
        "query": "test query",
        "query_type": "simple",
        "sub_queries": [],
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "refusal": False,
        "needs_clarification": False,
        "clarification_question": "",
        "retry_count": 0,
        "error": None,
        "validation_passed": False,
    }

    assert state["query"] == "test query"
    assert state["sub_queries"] == []
    assert state["retrieved_chunks"] == []
    assert state["answer"] == ""
    assert state["citations"] == []
    assert state["confidence"] == 0.0
    assert state["refusal"] is False
    assert state["needs_clarification"] is False
    assert state["clarification_question"] == ""
    assert state["retry_count"] == 0
    assert state["error"] is None


def test_agent_state_retrieved_chunks_accepts_chunk_dicts():
    from src.contracts import ChunkDict
    from src.schemas import AgentState

    chunk: ChunkDict = {
        "text": "firewall controls",
        "metadata": {"doc": "NIST SP 800-82", "section": "5.2", "page": 42},
        "score": 0.88,
    }

    state: AgentState = {
        "query": "test query",
        "query_type": "simple",
        "sub_queries": [],
        "retrieved_chunks": [chunk],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "refusal": False,
        "needs_clarification": False,
        "clarification_question": "",
        "retry_count": 0,
        "error": None,
        "validation_passed": False,
    }

    assert len(state["retrieved_chunks"]) == 1
    assert state["retrieved_chunks"][0]["text"] == "firewall controls"


def test_agentstate_defaults_has_all_required_keys():
    from src.schemas import AGENTSTATE_DEFAULTS, AgentState

    required_keys = AgentState.__annotations__.keys()
    for key in required_keys:
        assert key in AGENTSTATE_DEFAULTS, \
            f"AGENTSTATE_DEFAULTS missing key: {key}"
