import pytest


def test_gemini_client_init_stores_config():
    from src.llm import GeminiClient

    client = GeminiClient(api_key="fake-key", model="gemini-2.5-flash")

    assert client.api_key == "fake-key"
    assert client.model == "gemini-2.5-flash"


def test_gemini_client_default_model():
    from src.llm import GeminiClient

    client = GeminiClient(api_key="fake-key")

    assert client.model == "gemini-2.5-flash"


def test_gemini_generate_calls_api_with_temperature_zero(mocker):
    from src.llm import GeminiClient

    mock_genai = mocker.patch("src.llm.genai")
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.return_value.text = "test response"

    client = GeminiClient(api_key="fake-key")
    client.generate("system prompt", "user query")

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"]["temperature"] == 0.0


def test_gemini_generate_returns_response_text(mocker):
    from src.llm import GeminiClient

    mock_genai = mocker.patch("src.llm.genai")
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.return_value.text = "Hello answer"

    client = GeminiClient(api_key="fake-key")
    result = client.generate("system prompt", "user query")

    assert result == "Hello answer"


def test_gemini_generate_passes_max_tokens(mocker):
    from src.llm import GeminiClient

    mock_genai = mocker.patch("src.llm.genai")
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.return_value.text = "test response"

    client = GeminiClient(api_key="fake-key")
    client.generate("sys", "query", max_tokens=512)

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"]["max_output_tokens"] == 512


def test_exponential_backoff_sleep_durations(mocker):
    from src.llm import GeminiClient

    mock_genai = mocker.patch("src.llm.genai")
    mock_client = mock_genai.Client.return_value
    success_response = mocker.Mock()
    success_response.text = "test response"
    mock_client.models.generate_content.side_effect = [
        Exception("429 rate limit exceeded"),
        Exception("429 rate limit exceeded"),
        success_response,
    ]
    mock_sleep = mocker.patch("src.llm.time.sleep")

    client = GeminiClient(api_key="fake-key")
    result = client.generate("sys", "query")

    assert result == "test response"
    assert mock_sleep.call_count == 2

    first_sleep = mock_sleep.call_args_list[0][0][0]
    second_sleep = mock_sleep.call_args_list[1][0][0]
    assert 1.5 <= first_sleep <= 2.5
    assert 3.5 <= second_sleep <= 4.5


def test_max_retries_exceeded_after_three_attempts(mocker):
    from src.llm import GeminiClient, MaxRetriesExceeded

    mock_genai = mocker.patch("src.llm.genai")
    mock_client = mock_genai.Client.return_value
    mock_client.models.generate_content.side_effect = Exception("429 rate limit exceeded")
    mocker.patch("src.llm.time.sleep")

    client = GeminiClient(api_key="fake-key")

    with pytest.raises(MaxRetriesExceeded):
        client.generate("sys", "query")

    assert mock_client.models.generate_content.call_count == 3


def test_rate_limit_error_when_rpm_ceiling_exceeded(mocker):
    from src.llm import GeminiClient, RateLimitError

    mock_genai = mocker.patch("src.llm.genai")
    mock_client = mock_genai.Client.return_value
    success_response = mocker.Mock()
    success_response.text = "test response"
    mock_client.models.generate_content.return_value = success_response
    mocker.patch("src.llm.time.time", return_value=1000.0)

    client = GeminiClient(api_key="fake-key")

    for _ in range(15):
        client.generate("sys", "query")

    with pytest.raises(RateLimitError):
        client.generate("sys", "query")


def test_generate_json_strips_triple_backtick_json_fence(mocker):
    from src.llm import GeminiClient

    client = GeminiClient(api_key="fake-key")
    mocker.patch.object(client, "generate", return_value='```json\n{"key": "value"}\n```')

    result = client.generate_json("classification prompt")

    assert result == {"key": "value"}


def test_generate_json_strips_plain_triple_backtick_fence(mocker):
    from src.llm import GeminiClient

    client = GeminiClient(api_key="fake-key")
    mocker.patch.object(client, "generate", return_value='```\n{"key": "value"}\n```')

    result = client.generate_json("classification prompt")

    assert result == {"key": "value"}


def test_generate_json_handles_no_fence(mocker):
    from src.llm import GeminiClient

    client = GeminiClient(api_key="fake-key")
    mocker.patch.object(client, "generate", return_value='{"key": "value"}')

    result = client.generate_json("classification prompt")

    assert result == {"key": "value"}


def test_generate_json_raises_on_invalid_json(mocker):
    from src.llm import GeminiClient, JSONParseError

    client = GeminiClient(api_key="fake-key")
    mocker.patch.object(client, "generate", return_value="not valid json at all")

    with pytest.raises(JSONParseError):
        client.generate_json("classification prompt")


def test_huggingface_client_init_stores_config():
    from src.llm import HuggingFaceClient

    client = HuggingFaceClient(api_key="fake-hf-key")

    assert client.api_key == "fake-hf-key"
    assert "mistral" in client.model.lower()


def test_huggingface_generate_calls_inference_api(mocker):
    from src.llm import HuggingFaceClient

    mock_post = mocker.patch("src.llm.requests.post")
    mock_post.return_value.json.return_value = [{"generated_text": "HF answer"}]

    client = HuggingFaceClient(api_key="fake-hf-key")
    client.generate("system prompt", "user query")

    called_url = mock_post.call_args[0][0]
    assert "huggingface.co" in called_url


def test_huggingface_generate_returns_text(mocker):
    from src.llm import HuggingFaceClient

    mock_post = mocker.patch("src.llm.requests.post")
    mock_post.return_value.json.return_value = [{"generated_text": "HF answer"}]

    client = HuggingFaceClient(api_key="fake-hf-key")
    result = client.generate("system prompt", "user query")

    assert result == "HF answer"


def test_llm_router_init_stores_clients(mocker):
    from src.llm import LLMRouter

    primary = mocker.Mock()
    fallback = mocker.Mock()
    router = LLMRouter(primary=primary, fallback=fallback)

    assert router.primary is primary
    assert router.fallback is fallback


def test_llm_router_generate_uses_primary_on_success(mocker):
    from src.llm import LLMRouter

    primary = mocker.Mock()
    primary.generate.return_value = "primary response"
    fallback = mocker.Mock()
    router = LLMRouter(primary=primary, fallback=fallback)

    result = router.generate("sys", "query")

    assert result == "primary response"
    assert fallback.generate.call_count == 0


def test_llm_router_falls_back_on_max_retries_exceeded(mocker):
    from src.llm import LLMRouter, MaxRetriesExceeded

    primary = mocker.Mock()
    primary.generate.side_effect = MaxRetriesExceeded("rate limited")
    fallback = mocker.Mock()
    fallback.generate.return_value = "fallback response"
    router = LLMRouter(primary=primary, fallback=fallback)

    result = router.generate("sys", "query")

    assert result == "fallback response"


def test_llm_router_raises_all_providers_exhausted(mocker):
    from src.llm import AllProvidersExhausted, LLMRouter, MaxRetriesExceeded

    primary = mocker.Mock()
    primary.generate.side_effect = MaxRetriesExceeded("rate limited")
    fallback = mocker.Mock()
    fallback.generate.side_effect = Exception("HF also down")
    router = LLMRouter(primary=primary, fallback=fallback)

    with pytest.raises(AllProvidersExhausted):
        router.generate("sys", "query")


def test_llm_router_generate_json_uses_primary(mocker):
    from src.llm import LLMRouter

    primary = mocker.Mock()
    primary.generate_json.return_value = {"query_type": "simple"}
    fallback = mocker.Mock()
    router = LLMRouter(primary=primary, fallback=fallback)

    result = router.generate_json("classification prompt")

    assert result == {"query_type": "simple"}


def test_strip_markdown_json_fence_handles_whitespace_variations():
    from src.llm import _strip_markdown_json_fence

    assert _strip_markdown_json_fence(
        '```json\n  {"a": 1}  \n```'
    ).strip() == '{"a": 1}'
    assert _strip_markdown_json_fence('{"a": 1}') == '{"a": 1}'
