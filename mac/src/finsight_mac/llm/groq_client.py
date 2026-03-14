"""Groq cloud LLM client — fallback when Ollama times out.

Uses Groq's OpenAI-compatible API for fast cloud inference.
Free tier: 14,400 requests/day, sufficient for tree generation.
Get an API key at https://console.groq.com/keys
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)


class GroqClient:
    """Sync and async Groq client via OpenAI-compatible API.

    Drop-in replacement for OllamaClient when Ollama times out
    on large documents. Uses Groq's blazing-fast cloud inference.

    Example:
        >>> client = GroqClient()
        >>> response = client.chat("Summarize this SEC filing section.")
        >>> data = await client.achat_json(system_prompt, user_prompt)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.groq_model
        self.api_key = api_key or settings.groq_api_key
        self.base_url = base_url or settings.groq_base_url
        self.temperature = temperature if temperature is not None else settings.temperature

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required for Groq cloud inference. "
                "Get a free key at https://console.groq.com/keys"
            )

        # Initialize clients (OpenAI-compatible)
        self._sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120,  # Groq is fast, 120s is generous
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120,
        )

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Synchronous chat completion via Groq.

        Args:
            user_message: The user's message/query.
            system_message: Optional system prompt.
            temperature: Override default temperature.

        Returns:
            The model's response text.
        """
        messages = self._build_messages(system_message, user_message)
        temp = temperature if temperature is not None else self.temperature

        start = time.time()
        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
        )
        elapsed = time.time() - start

        content = response.choices[0].message.content or ""
        tokens = response.usage
        if tokens:
            tok_per_sec = tokens.completion_tokens / max(elapsed, 0.01)
            logger.info(
                f"[Groq] {self.model}: {tokens.completion_tokens} tokens "
                f"in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)"
            )

        return content

    async def achat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Asynchronous chat completion via Groq.

        Args:
            user_message: The user's message/query.
            system_message: Optional system prompt.
            temperature: Override default temperature.

        Returns:
            The model's response text.
        """
        messages = self._build_messages(system_message, user_message)
        temp = temperature if temperature is not None else self.temperature

        start = time.time()
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
        )
        elapsed = time.time() - start

        content = response.choices[0].message.content or ""
        tokens = response.usage
        if tokens:
            tok_per_sec = tokens.completion_tokens / max(elapsed, 0.01)
            logger.info(
                f"[Groq] {self.model}: {tokens.completion_tokens} tokens "
                f"in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)"
            )

        return content

    async def achat_json(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Async chat completion expecting JSON response.

        Parses the response as JSON, handling common LLM formats
        (markdown code blocks, thinking tags, etc.).

        Returns:
            Parsed JSON dict.

        Raises:
            json.JSONDecodeError: If response cannot be parsed as JSON.
        """
        raw = await self.achat(user_message, system_message, temperature)
        return self._parse_json_response(raw)

    def chat_json(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Sync chat completion expecting JSON response."""
        raw = self.chat(user_message, system_message, temperature)
        return self._parse_json_response(raw)

    @staticmethod
    def _build_messages(system_message: str | None, user_message: str) -> list[dict[str, str]]:
        """Build the messages list for the API call."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common formats.

        Handles:
        - Raw JSON
        - JSON wrapped in ```json ... ``` code blocks
        - JSON with trailing commas
        - Thinking tags <think>...</think> before JSON
        """
        text = raw.strip()

        # Remove thinking blocks (Qwen/DeepSeek style)
        if "<think>" in text:
            think_end = text.rfind("</think>")
            if think_end >= 0:
                text = text[think_end + len("</think>") :].strip()

        # Remove markdown code block wrapper
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        # Handle trailing commas (common LLM mistake)
        text = re.sub(r",\s*([\]}])", r"\1", text)

        return json.loads(text)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Groq is configured with an API key."""
        settings = get_settings()
        return bool(settings.groq_api_key)

    async def check_health(self) -> bool:
        """Check if Groq API is reachable and key is valid."""
        try:
            response = await self.achat("Say 'ok'.", temperature=0)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Groq health check failed: {e}")
            return False
