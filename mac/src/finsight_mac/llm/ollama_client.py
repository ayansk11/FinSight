"""Ollama LLM client — async OpenAI-compatible wrapper.

Connects to Ollama's OpenAI-compatible API at localhost:11434/v1.
Provides both sync and async interfaces for LLM completions.
Supports structured JSON output for agent responses.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI

from finsight_mac.config import get_settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Synchronous and asynchronous Ollama client via OpenAI-compatible API.

    Example:
        >>> client = OllamaClient()
        >>> response = client.chat("What is Apple's revenue?")
        >>> print(response)

        >>> # Async usage
        >>> response = await client.achat("What is Apple's revenue?")

        >>> # Structured JSON output
        >>> data = await client.achat_json(system_prompt, user_prompt)
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_openai_url
        self.temperature = temperature if temperature is not None else settings.temperature
        self.timeout = settings.ollama_timeout

        # Initialize clients
        self._sync_client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",  # Ollama doesn't need a real key
            timeout=self.timeout,
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="ollama",
            timeout=self.timeout,
        )

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Synchronous chat completion.

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
                f"LLM response: {tokens.completion_tokens} tokens in {elapsed:.1f}s "
                f"({tok_per_sec:.1f} tok/s)"
            )

        return content

    async def achat(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Asynchronous chat completion.

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
                f"LLM response: {tokens.completion_tokens} tokens in {elapsed:.1f}s "
                f"({tok_per_sec:.1f} tok/s)"
            )

        return content

    async def achat_json(
        self,
        user_message: str,
        system_message: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Async chat completion expecting JSON response.

        Parses the response as JSON, handling common LLM response formats
        (e.g., markdown code blocks around JSON).

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
        - Qwen3.5 thinking tags <think>...</think> before JSON
        """
        text = raw.strip()

        # Remove Qwen3.5 thinking blocks
        if "<think>" in text:
            think_end = text.rfind("</think>")
            if think_end >= 0:
                text = text[think_end + len("</think>") :].strip()

        # Remove markdown code block wrapper
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        # Handle trailing commas (common LLM mistake)
        import re

        text = re.sub(r",\s*([\]}])", r"\1", text)

        return json.loads(text)

    async def check_health(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = await self.achat("Say 'ok'.", temperature=0)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
