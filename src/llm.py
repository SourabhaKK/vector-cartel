"""
llm.py — LLM client layer for SecureOps Assistant.

Three classes:
  GeminiClient      — primary LLM, Gemini 2.5 Flash, free tier
  HuggingFaceClient — fallback LLM, Mistral-7B-Instruct
  LLMRouter         — orchestrates primary-then-fallback chain

SDK NOTE:
Uses the current google-genai SDK (genai.Client(...).models.generate_content(...)),
not the older, now-deprecated google-generativeai SDK
(genai.configure(...) + genai.GenerativeModel(...)). The official
hackathon starter notebook already uses google-genai + gemini-2.5-flash;
this module was migrated to match.

RATE LIMIT DESIGN:
Gemini free tier allows 15 requests per minute. GeminiClient
tracks call timestamps internally and raises RateLimitError
before attempting a call that would exceed this ceiling.
This is a client-side guard, not a substitute for server-side
rate limit handling — the exponential backoff handles 429
responses from Gemini's API independently.

FALLBACK DESIGN:
LLMRouter tries GeminiClient first. If GeminiClient raises
MaxRetriesExceeded (meaning the backoff retries were exhausted),
LLMRouter falls back to HuggingFaceClient. If both fail,
AllProvidersExhausted is raised and the caller (agent.py) is
responsible for returning a graceful error to the user rather
than crashing.

USED BY:
  src/agent.py — every node that needs an LLM call uses
  LLMRouter, never GeminiClient or HuggingFaceClient directly.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict, Optional

from google import genai
import requests

logger = logging.getLogger(__name__)


class SecureOpsLLMError(Exception):
    """Base exception for all LLM-related errors in this module."""

    pass


class RateLimitError(SecureOpsLLMError):
    """Raised when the RPM ceiling is exceeded."""

    pass


class MaxRetriesExceeded(SecureOpsLLMError):
    """Raised when all retry attempts have been exhausted."""

    pass


class JSONParseError(SecureOpsLLMError):
    """Raised when LLM output cannot be parsed as valid JSON."""

    pass


class AllProvidersExhausted(SecureOpsLLMError):
    """Raised when both primary and fallback LLM providers fail."""

    pass


def _strip_markdown_json_fence(text: str) -> str:
    """
    Strips ```json...``` or ```...``` fences from LLM output.
    Returns the text unchanged if no fence is present.

    Handles all three cases tested:
      - ```json\n{...}\n```
      - ```\n{...}\n```
      - {...} with no fence at all
    """
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self._call_timestamps: list[float] = []
        self._rpm_limit = 15

    def generate(
        self, system_prompt: str, user_query: str, max_tokens: int = 1024
    ) -> str:
        """
        Generates a response from Gemini 2.5 Flash.

        Args:
            system_prompt: The full system prompt (rules + context).
            user_query: The user's question.
            max_tokens: Maximum output tokens, default 1024.

        Returns:
            The generated response text.

        Raises:
            RateLimitError: If calling now would exceed 15 RPM.
            MaxRetriesExceeded: If 3 retry attempts all fail.

        Called by: src/agent.py synthesize_answer node, via LLMRouter.
        """
        self._check_rate_limit()

        contents = f"{system_prompt}\n\n{user_query}"

        for attempt in range(1, 4):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={
                        "temperature": 0.0,
                        "max_output_tokens": max_tokens,
                        # gemini-2.5-flash's "thinking" feature consumes
                        # output tokens on internal reasoning by default
                        # (verified: ~980 of a 1024 budget went to
                        # thoughts_token_count, leaving the answer cut off
                        # mid-sentence with finish_reason=MAX_TOKENS).
                        # Disabling it keeps the full max_output_tokens
                        # budget for the actual visible answer.
                        "thinking_config": {"thinking_budget": 0},
                    },
                )
                self._call_timestamps.append(time.time())
                return response.text
            except Exception:
                if attempt < 3:
                    time.sleep(self._backoff_sleep_duration(attempt))
                    continue
                raise MaxRetriesExceeded(
                    "Exhausted all retry attempts calling Gemini"
                )

    def _check_rate_limit(self) -> None:
        """
        Raises RateLimitError if calling now would exceed the
        15 RPM free tier ceiling. Prunes timestamps older than
        60 seconds before checking.
        """
        now = time.time()
        self._call_timestamps = [
            t for t in self._call_timestamps if t > now - 60
        ]
        if len(self._call_timestamps) >= self._rpm_limit:
            raise RateLimitError("RPM ceiling exceeded")

    def _backoff_sleep_duration(self, attempt: int) -> float:
        """
        Calculates sleep duration before retry attempt N.

        Formula: 2^attempt seconds, with ±0.5s jitter to avoid
        thundering herd if multiple requests retry simultaneously.

        Args:
            attempt: The retry attempt number (1, 2, or 3).

        Returns:
            Sleep duration in seconds.
        """
        return (2**attempt) + random.uniform(-0.5, 0.5)

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generates a response from Gemini and parses it as JSON.

        Args:
            prompt: The full prompt (e.g. classification or
                decomposition prompt) instructing the model to
                return JSON only.

        Returns:
            The parsed JSON response as a dict.

        Raises:
            RateLimitError: If calling now would exceed 15 RPM.
            MaxRetriesExceeded: If 3 retry attempts all fail.
            JSONParseError: If the response is not valid JSON
                after stripping markdown fences.

        Called by: src/agent.py classify_query and decompose_query
        nodes, via LLMRouter.
        """
        raw = self.generate(prompt, "")
        cleaned = _strip_markdown_json_fence(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise JSONParseError(f"Could not parse LLM output as JSON: {e}")


class HuggingFaceClient:
    def __init__(
        self, api_key: str, model: str = "Qwen/Qwen2.5-7B-Instruct"
    ) -> None:
        self.api_key = api_key
        self.model = model

    def generate(
        self, system_prompt: str, user_query: str, max_tokens: int = 1024
    ) -> str:
        """
        Generates a response from the HuggingFace Inference API.

        Args:
            system_prompt: The full system prompt (rules + context).
            user_query: The user's question.
            max_tokens: Maximum output tokens, default 1024.

        Returns:
            The generated response text.

        Raises:
            Exception: Propagates any request or parsing failure
                from the HuggingFace Inference API unchanged —
                LLMRouter is responsible for catching these.

        Called by: src/agent.py nodes, via LLMRouter as the
        fallback provider when GeminiClient raises
        MaxRetriesExceeded.

        URL/PAYLOAD/MODEL NOTE: HuggingFace decommissioned
        api-inference.huggingface.co entirely (verified: DNS resolution
        fails for that hostname). router.huggingface.co is the current
        routing endpoint. The classic {"inputs": ...} task-based payload
        only works for models still mapped to HF's own "hf-inference"
        provider -- almost nothing is anymore; models are now served via
        third-party providers (featherless-ai, together, novita, ...)
        on the "conversational" task, reachable only via the
        OpenAI-compatible /v1/chat/completions shape used here.

        The original default model (mistralai/Mistral-7B-Instruct-v0.1,
        served only via featherless-ai) returned a real 400
        "not supported by any provider you have enabled" with a real
        key -- that provider wasn't enabled for the account tested.
        Qwen/Qwen2.5-7B-Instruct (served via the "together" provider)
        was verified end-to-end with a real key and a real call through
        this exact method: HTTP 200, correct response text extracted.
        Provider enablement is per-account, so if this breaks again,
        check https://huggingface.co/settings/inference-providers and/or
        pick another model from its inferenceProviderMapping.
        """
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            "max_tokens": max_tokens,
        }

        response = requests.post(url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]


class LLMRouter:
    def __init__(self, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback

    def generate(self, system_prompt: str, user_query: str) -> str:
        """
        Generates a response, trying primary then falling back.

        Args:
            system_prompt: The full system prompt (rules + context).
            user_query: The user's question.

        Returns:
            The generated response text from whichever provider
            succeeded.

        Raises:
            AllProvidersExhausted: If primary raises
                MaxRetriesExceeded and fallback also fails.

        Called by: every src/agent.py node that needs free-text
        LLM generation.
        """
        try:
            return self.primary.generate(system_prompt, user_query)
        except MaxRetriesExceeded:
            logger.warning("Primary LLM exhausted, falling back to HuggingFace")
            try:
                return self.fallback.generate(system_prompt, user_query)
            except Exception as e:
                raise AllProvidersExhausted(f"Both providers failed: {e}")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generates a JSON response using the primary provider only.

        Args:
            prompt: The full prompt instructing the model to
                return JSON only.

        Returns:
            The parsed JSON response as a dict.

        Raises:
            AllProvidersExhausted: If primary raises
                MaxRetriesExceeded. There is no JSON-capable
                fallback in this minimal version.

        Called by: src/agent.py classify_query and decompose_query
        nodes.
        """
        try:
            return self.primary.generate_json(prompt)
        except MaxRetriesExceeded:
            logger.warning("Primary LLM exhausted for JSON call")
            raise AllProvidersExhausted("Primary failed, no JSON fallback")
