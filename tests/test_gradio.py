def test_format_response_with_citations():
    from src.gradio_demo import format_response
    from src.schemas import Citation

    citations = [
        Citation(
            doc="NIST SP 800-82 Rev 3",
            section="5.2.3",
            page=42,
            snippet="firewall segmentation controls",
        )
    ]

    result = format_response(
        answer="Firewalls should segment OT networks.",
        citations=citations,
        refusal=False,
        needs_clarification=False,
        clarification_question="",
    )

    assert "Firewalls should segment OT networks." in result
    assert "NIST SP 800-82 Rev 3" in result
    assert "5.2.3" in result
    assert "Sources:" in result


def test_format_response_with_multiple_citations_numbered():
    from src.gradio_demo import format_response
    from src.schemas import Citation

    citations = [
        Citation(doc="NIST SP 800-82 Rev 3", section="5.2", snippet="text one"),
        Citation(
            doc="CISA Advisory ICSA-24-001",
            section="Affected Products",
            snippet="text two",
        ),
    ]

    result = format_response(
        answer="Combined answer.",
        citations=citations,
        refusal=False,
        needs_clarification=False,
        clarification_question="",
    )

    assert "[1]" in result
    assert "[2]" in result
    assert "NIST SP 800-82 Rev 3" in result
    assert "CISA Advisory ICSA-24-001" in result


def test_format_response_refusal_has_no_sources_block():
    from src.gradio_demo import format_response

    result = format_response(
        answer="I don't have enough information in the corpus to answer this question.",
        citations=[],
        refusal=True,
        needs_clarification=False,
        clarification_question="",
    )

    assert "I don't have enough information" in result
    assert "Sources:" not in result


def test_format_response_clarification_returns_question_only():
    from src.gradio_demo import format_response

    result = format_response(
        answer="",
        citations=[],
        refusal=False,
        needs_clarification=True,
        clarification_question="Which Siemens product line are you asking about?",
    )

    assert "Which Siemens product line are you asking about?" in result
    assert "Sources:" not in result


def test_format_response_handles_empty_citations_gracefully():
    from src.gradio_demo import format_response

    result = format_response(
        answer="Some answer with no sources.",
        citations=[],
        refusal=False,
        needs_clarification=False,
        clarification_question="",
    )

    assert "Some answer with no sources." in result
    assert "Sources:" not in result


def test_format_response_exception_returns_error_string():
    from src.gradio_demo import format_response

    result = format_response(
        answer="test",
        citations=None,
        refusal=False,
        needs_clarification=False,
        clarification_question="",
    )

    result_lower = result.lower()
    assert "error" in result_lower


def test_setup_reads_gemini_api_key_from_env(mocker, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("HF_API_KEY", "fake-hf-key")
    mocker.patch("src.gradio_demo.GeminiClient")
    mocker.patch("src.gradio_demo.HuggingFaceClient")

    from src.gradio_demo import setup

    llm_router, retrieval_fn = setup()

    assert llm_router is not None
    assert callable(retrieval_fn)


def test_setup_falls_back_to_stub_retrieval_when_rag_layer_missing(
    mocker, monkeypatch
):
    """
    Forces ImportError on `from src.retrieval import build_retrieval_fn`
    via the standard sys.modules-entry-set-to-None trick, rather than
    relying on src/retrieval.py being ambiently absent. Now that
    rag-layer has merged and src/retrieval.py genuinely exists, this
    is the only deterministic way to exercise setup()'s fallback path
    -- without this, the test would (correctly) pick up the real
    build_retrieval_fn and fail, since the scenario it's meant to
    simulate ("rag-layer not merged") no longer reflects reality.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("HF_API_KEY", "fake-hf-key")
    mocker.patch("src.gradio_demo.GeminiClient")
    mocker.patch("src.gradio_demo.HuggingFaceClient")
    mocker.patch.dict("sys.modules", {"src.retrieval": None})

    from src.gradio_demo import setup

    llm_router, retrieval_fn = setup()
    result = retrieval_fn("any query")

    assert result == []


def test_chat_fn_calls_run_agent_and_formats_response(mocker):
    from src.gradio_demo import chat_fn
    from src.schemas import Citation, SecureOpsAnswer

    mock_run_agent = mocker.patch("src.gradio_demo.run_agent")
    mock_run_agent.return_value = SecureOpsAnswer(
        answer="Test answer [Source: NIST SP 800-82 Rev 3 | 5.2]",
        citations=[
            Citation(doc="NIST SP 800-82 Rev 3", section="5.2", snippet="test")
        ],
        confidence=0.9,
        refusal=False,
        query_type="simple",
        sources_used=["NIST SP 800-82 Rev 3"],
    )
    mock_llm_router = mocker.Mock()
    mock_retrieval_fn = mocker.Mock()

    result = chat_fn("test message", [], mock_llm_router, mock_retrieval_fn)

    assert "Test answer" in result
    assert "NIST SP 800-82 Rev 3" in result


def test_chat_fn_returns_error_string_on_exception(mocker):
    from src.gradio_demo import chat_fn

    mocker.patch(
        "src.gradio_demo.run_agent", side_effect=Exception("something broke")
    )
    mock_llm_router = mocker.Mock()
    mock_retrieval_fn = mocker.Mock()

    result = chat_fn("test message", [], mock_llm_router, mock_retrieval_fn)

    result_lower = result.lower()
    assert "error" in result_lower
    assert "please retry" in result_lower or "retry" in result_lower


def test_format_citation_list_single_citation():
    from src.gradio_demo import _format_citation_list
    from src.schemas import Citation

    citations = [
        Citation(doc="NIST SP 800-82 Rev 3", section="5.2", snippet="test")
    ]

    result = _format_citation_list(citations)

    assert "[1]" in result
    assert "NIST SP 800-82 Rev 3" in result


def test_get_required_env_raises_when_missing(monkeypatch):
    from src.gradio_demo import _get_required_env

    monkeypatch.delenv("SOME_NONEXISTENT_KEY", raising=False)

    import pytest

    with pytest.raises(EnvironmentError):
        _get_required_env("SOME_NONEXISTENT_KEY")
