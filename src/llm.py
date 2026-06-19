from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict, Optional

import google.generativeai as genai
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


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
        genai.configure(api_key=api_key)
        self._call_timestamps: list[float] = []
        self._rpm_limit = 15

    def generate(
        self, system_prompt: str, user_query: str, max_tokens: int = 1024
    ) -> str:
        now = time.time()
        recent_calls = [t for t in self._call_timestamps if t > now - 60]
        if len(recent_calls) >= self._rpm_limit:
            raise RateLimitError("RPM ceiling exceeded")

        contents = f"{system_prompt}\n\n{user_query}"

        for attempt in range(1, 4):
            try:
                model = genai.GenerativeModel(
                    self.model,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": max_tokens,
                    },
                )
                response = model.generate_content(contents)
                self._call_timestamps.append(time.time())
                return response.text
            except Exception:
                if attempt < 3:
                    sleep_time = (2**attempt) + random.uniform(-0.5, 0.5)
                    time.sleep(sleep_time)
                    continue
                raise MaxRetriesExceeded(
                    "Exhausted all retry attempts calling Gemini"
                )

    def generate_json(self, prompt: str) -> dict:
        raw = self.generate(prompt, "")
        cleaned = _strip_markdown_fence(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise JSONParseError(f"Could not parse LLM output as JSON: {e}")


class HuggingFaceClient:
    def __init__(
        self, api_key: str, model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    ):
        self.api_key = api_key
        self.model = model

    def generate(
        self, system_prompt: str, user_query: str, max_tokens: int = 1024
    ) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": f"{system_prompt}\n\n{user_query}"}

        response = requests.post(url, headers=headers, json=payload)
        return response.json()[0]["generated_text"]


class LLMRouter:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def generate(self, system_prompt: str, user_query: str) -> str:
        try:
            return self.primary.generate(system_prompt, user_query)
        except MaxRetriesExceeded:
            logger.warning("Primary LLM exhausted, falling back to HuggingFace")
            try:
                return self.fallback.generate(system_prompt, user_query)
            except Exception as e:
                raise AllProvidersExhausted(f"Both providers failed: {e}")

    def generate_json(self, prompt: str) -> dict:
        try:
            return self.primary.generate_json(prompt)
        except MaxRetriesExceeded:
            logger.warning("Primary LLM exhausted for JSON call")
            raise AllProvidersExhausted("Primary failed, no JSON fallback")
