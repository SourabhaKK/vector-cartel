from src.schemas import AGENTSTATE_DEFAULTS


def make_state(**overrides):
    state = dict(AGENTSTATE_DEFAULTS)
    state.update(overrides)
    return state


NIST_CHUNK = {
    "text": "Firewalls should segment OT networks from IT networks.",
    "metadata": {
        "doc": "NIST SP 800-82 Rev 3",
        "section": "5.2.3",
        "page": 87,
    },
    "score": 0.91,
}


# ── NODE 1 — classify_query ──────────────────────────────────────


def test_classify_query_returns_simple(mocker):
    from src.agent import classify_query

    mock_llm = mocker.Mock()
    mock_llm.generate_json.return_value = {
        "query_type": "simple",
        "confidence": 0.9,
    }
    state = make_state(query="What does NIST say about VPNs?")

    result = classify_query(state, mock_llm)

    assert result == {"query_type": "simple"}


def test_classify_query_returns_multi_hop(mocker):
    from src.agent import classify_query

    mock_llm = mocker.Mock()
    mock_llm.generate_json.return_value = {"query_type": "multi_hop"}
    state = make_state(query="What does NIST say about VPNs?")

    result = classify_query(state, mock_llm)

    assert result == {"query_type": "multi_hop"}


def test_classify_query_defaults_to_simple_on_json_parse_error(mocker):
    from src.agent import classify_query
    from src.llm import JSONParseError

    mock_llm = mocker.Mock()
    mock_llm.generate_json.side_effect = JSONParseError("bad json")
    state = make_state(query="What does NIST say about VPNs?")

    result = classify_query(state, mock_llm)

    assert result == {"query_type": "simple"}


def test_classify_query_passes_query_to_prompt_builder(mocker):
    from src.agent import classify_query

    mock_build_prompt = mocker.patch(
        "src.prompts.build_classification_prompt", return_value="prompt text"
    )
    mock_llm = mocker.Mock()
    mock_llm.generate_json.return_value = {"query_type": "simple"}
    state = make_state(query="specific test query")

    classify_query(state, mock_llm)

    mock_build_prompt.assert_called_with("specific test query")


# ── NODE 2 — decompose_query ──────────────────────────────────────


def test_decompose_query_returns_two_sub_queries(mocker):
    from src.agent import decompose_query

    mock_llm = mocker.Mock()
    mock_llm.generate_json.return_value = {
        "sub_queries": ["NIST query", "CISA query"]
    }
    state = make_state(query="complex query")

    result = decompose_query(state, mock_llm)

    assert result == {"sub_queries": ["NIST query", "CISA query"]}


def test_decompose_query_falls_back_on_empty_list(mocker):
    from src.agent import decompose_query

    mock_llm = mocker.Mock()
    mock_llm.generate_json.return_value = {"sub_queries": []}
    state = make_state(query="original query")

    result = decompose_query(state, mock_llm)

    assert result == {"sub_queries": ["original query"]}


def test_decompose_query_falls_back_on_json_parse_error(mocker):
    from src.agent import decompose_query
    from src.llm import JSONParseError

    mock_llm = mocker.Mock()
    mock_llm.generate_json.side_effect = JSONParseError("bad json")
    state = make_state(query="original query")

    result = decompose_query(state, mock_llm)

    assert result == {"sub_queries": ["original query"]}


def test_decompose_query_caps_at_three_sub_queries(mocker):
    from src.agent import decompose_query

    mock_llm = mocker.Mock()
    mock_llm.generate_json.return_value = {
        "sub_queries": ["q1", "q2", "q3", "q4", "q5"]
    }
    state = make_state(query="complex query")

    result = decompose_query(state, mock_llm)

    assert len(result["sub_queries"]) == 3


# ── NODE 3 — handle_refusal ───────────────────────────────────────


def test_handle_refusal_sets_refusal_true():
    from src.agent import handle_refusal

    state = make_state(query="What is our firewall config?")

    result = handle_refusal(state)

    assert result["refusal"] is True


def test_handle_refusal_sets_confidence_zero():
    from src.agent import handle_refusal

    state = make_state(query="What is our firewall config?")

    result = handle_refusal(state)

    assert result["confidence"] == 0.0


def test_handle_refusal_answer_contains_standard_message():
    from src.agent import handle_refusal

    state = make_state(query="What is our firewall config?")

    result = handle_refusal(state)

    assert "I don't have enough information in the corpus" in result["answer"]


# ── NODE 4 — handle_clarification ─────────────────────────────────


def test_handle_clarification_sets_needs_clarification_true(mocker):
    from src.agent import handle_clarification

    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = "Which Siemens product line?"
    state = make_state(query="Tell me about Siemens vulnerabilities")

    result = handle_clarification(state, mock_llm)

    assert result["needs_clarification"] is True


def test_handle_clarification_sets_question_from_llm(mocker):
    from src.agent import handle_clarification

    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = "Which Siemens product line?"
    state = make_state(query="Tell me about Siemens vulnerabilities")

    result = handle_clarification(state, mock_llm)

    assert result["clarification_question"] == "Which Siemens product line?"


# ── NODE 5 — synthesize_answer ────────────────────────────────────


def test_synthesize_answer_calls_build_system_prompt_with_chunks(mocker):
    from src.agent import synthesize_answer

    mock_build_prompt = mocker.patch(
        "src.prompts.build_system_prompt", return_value="system prompt text"
    )
    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = (
        "Firewalls segment networks [Source: NIST SP 800-82 Rev 3 | 5.2.3]"
    )
    state = make_state(
        query="What does NIST say about firewalls?",
        retrieved_chunks=[NIST_CHUNK],
    )

    synthesize_answer(state, mock_llm)

    mock_build_prompt.assert_called_with([NIST_CHUNK])


def test_synthesize_answer_sets_answer_text(mocker):
    from src.agent import synthesize_answer

    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = (
        "Firewalls segment networks [Source: NIST SP 800-82 Rev 3 | 5.2.3]"
    )
    state = make_state(retrieved_chunks=[NIST_CHUNK])

    result = synthesize_answer(state, mock_llm)

    assert result["answer"] == mock_llm.generate.return_value


def test_synthesize_answer_parses_single_citation(mocker):
    from src.agent import synthesize_answer

    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = (
        "Firewalls segment networks [Source: NIST SP 800-82 Rev 3 | 5.2.3]"
    )
    state = make_state(retrieved_chunks=[NIST_CHUNK])

    result = synthesize_answer(state, mock_llm)

    assert len(result["citations"]) == 1
    assert result["citations"][0].doc == "NIST SP 800-82 Rev 3"
    assert result["citations"][0].section == "5.2.3"


def test_synthesize_answer_parses_multiple_citations(mocker):
    from src.agent import synthesize_answer

    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = (
        "Firewalls segment networks [Source: NIST SP 800-82 Rev 3 | 5.2.3]. "
        "Siemens devices are affected [Source: CISA Advisory ICSA-24-001 | "
        "Affected Products]."
    )
    state = make_state(retrieved_chunks=[NIST_CHUNK])

    result = synthesize_answer(state, mock_llm)

    assert len(result["citations"]) == 2


def test_synthesize_answer_handles_zero_citations(mocker):
    from src.agent import synthesize_answer

    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = (
        "I don't have enough information in the corpus to answer this question."
    )
    state = make_state(retrieved_chunks=[])

    result = synthesize_answer(state, mock_llm)

    assert result["citations"] == []


def test_synthesize_answer_confidence_reflects_chunk_scores(mocker):
    from src.agent import synthesize_answer

    mock_llm = mocker.Mock()
    mock_llm.generate.return_value = (
        "answer [Source: NIST SP 800-82 Rev 3 | 5.2.3]"
    )
    state = make_state(retrieved_chunks=[NIST_CHUNK])

    result = synthesize_answer(state, mock_llm)

    assert result["confidence"] == NIST_CHUNK["score"]


# ── NODE 6 — validate_citations ───────────────────────────────────


def test_validate_citations_passes_when_overlap_sufficient():
    from src.agent import validate_citations

    state = make_state(
        answer="Firewalls should segment OT networks",
        retrieved_chunks=[NIST_CHUNK],
        retry_count=0,
    )

    result = validate_citations(state)

    assert result["validation_passed"] is True


def test_validate_citations_fails_when_overlap_insufficient():
    from src.agent import validate_citations

    state = make_state(
        answer="Completely unrelated fabricated claim about quantum computing",
        retrieved_chunks=[NIST_CHUNK],
        retry_count=0,
    )

    result = validate_citations(state)

    assert result["validation_passed"] is False
    assert result["retry_count"] == 1


def test_validate_citations_passes_at_retry_limit_even_if_failed():
    from src.agent import validate_citations

    state = make_state(
        answer="Completely unrelated fabricated claim about quantum computing",
        retrieved_chunks=[NIST_CHUNK],
        retry_count=1,
    )

    result = validate_citations(state)

    assert result["validation_passed"] is True


def test_validate_citations_skips_check_for_refusal_answers():
    from src.agent import validate_citations

    state = make_state(
        answer="I don't have enough information in the corpus to answer this question.",
        retrieved_chunks=[],
        retry_count=0,
    )

    result = validate_citations(state)

    assert result["validation_passed"] is True


# ── ROUTE FUNCTIONS ────────────────────────────────────────────────


def test_route_by_query_type_simple():
    from src.agent import route_by_query_type

    state = make_state(query_type="simple")

    assert route_by_query_type(state) == "simple"


def test_route_by_query_type_multi_hop():
    from src.agent import route_by_query_type

    state = make_state(query_type="multi_hop")

    assert route_by_query_type(state) == "multi_hop"


def test_route_by_query_type_out_of_scope():
    from src.agent import route_by_query_type

    state = make_state(query_type="out_of_scope")

    assert route_by_query_type(state) == "out_of_scope"


def test_route_by_query_type_ambiguous():
    from src.agent import route_by_query_type

    state = make_state(query_type="ambiguous")

    assert route_by_query_type(state) == "ambiguous"


def test_route_by_validation_retry_when_failed_and_under_limit():
    from src.agent import route_by_validation

    state = make_state(validation_passed=False, retry_count=0)

    assert route_by_validation(state) == "retry"


def test_route_by_validation_passed_when_failed_but_at_limit():
    from src.agent import route_by_validation

    state = make_state(validation_passed=False, retry_count=1)

    assert route_by_validation(state) == "passed"


def test_route_by_validation_passed_when_validation_true():
    from src.agent import route_by_validation

    state = make_state(validation_passed=True, retry_count=0)

    assert route_by_validation(state) == "passed"
