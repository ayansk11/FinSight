"""Tests for the Groq cloud LLM client.

All API calls are mocked to ensure fast, deterministic tests.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_P = "finsight_mac.llm.groq_client"


def _mock_settings(**overrides):
    """Create a mock settings object with Groq defaults."""
    defaults = {
        "groq_api_key": "test_key",
        "groq_model": "llama-3.3-70b-versatile",
        "groq_base_url": "https://api.groq.com/openai/v1",
        "temperature": 0.1,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _mock_completion(content: str, completion_tokens: int = 50):
    """Create a mock OpenAI-style completion response."""
    usage = MagicMock()
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestGroqClientInit:
    """Tests for GroqClient initialization."""

    def test_no_api_key_raises(self):
        """GroqClient raises ValueError when no API key is configured."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings(groq_api_key=None)):
            from finsight_mac.llm.groq_client import GroqClient

            with pytest.raises(ValueError, match="GROQ_API_KEY is required"):
                GroqClient()

    def test_is_available_true(self):
        """is_available returns True when API key is set."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            from finsight_mac.llm.groq_client import GroqClient

            assert GroqClient.is_available() is True

    def test_is_available_false(self):
        """is_available returns False when no API key."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings(groq_api_key=None)):
            from finsight_mac.llm.groq_client import GroqClient

            assert GroqClient.is_available() is False

    def test_custom_params_override_settings(self):
        """Constructor params override settings values."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI"), patch(f"{_P}.AsyncOpenAI"):
                from finsight_mac.llm.groq_client import GroqClient

                client = GroqClient(
                    model="custom-model",
                    api_key="custom_key",
                    base_url="https://custom.api/v1",
                    temperature=0.5,
                )
                assert client.model == "custom-model"
                assert client.api_key == "custom_key"
                assert client.base_url == "https://custom.api/v1"
                assert client.temperature == 0.5


class TestGroqClientChat:
    """Tests for sync/async chat methods."""

    def test_chat_returns_content(self):
        """chat() returns the model's response text."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI") as mock_openai, patch(f"{_P}.AsyncOpenAI"):
                from finsight_mac.llm.groq_client import GroqClient

                mock_sync = MagicMock()
                mock_sync.chat.completions.create.return_value = _mock_completion(
                    "Hello from Groq!"
                )
                mock_openai.return_value = mock_sync

                client = GroqClient()
                result = client.chat("Say hello")
                assert result == "Hello from Groq!"

    def test_achat_returns_content(self):
        """achat() returns the model's async response text."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI"), patch(f"{_P}.AsyncOpenAI") as mock_async_openai:
                from finsight_mac.llm.groq_client import GroqClient

                mock_async = MagicMock()
                mock_async.chat.completions.create = AsyncMock(
                    return_value=_mock_completion("Async hello!")
                )
                mock_async_openai.return_value = mock_async

                client = GroqClient()
                result = asyncio.get_event_loop().run_until_complete(client.achat("Say hello"))
                assert result == "Async hello!"

    def test_chat_with_system_message(self):
        """chat() includes system message when provided."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI") as mock_openai, patch(f"{_P}.AsyncOpenAI"):
                from finsight_mac.llm.groq_client import GroqClient

                mock_sync = MagicMock()
                mock_sync.chat.completions.create.return_value = _mock_completion("ok")
                mock_openai.return_value = mock_sync

                client = GroqClient()
                client.chat("user msg", system_message="system msg")

                call_args = mock_sync.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                assert messages[0] == {"role": "system", "content": "system msg"}
                assert messages[1] == {"role": "user", "content": "user msg"}


class TestGroqJsonParsing:
    """Tests for JSON response parsing."""

    def test_parse_raw_json(self):
        """Parses raw JSON string."""
        from finsight_mac.llm.groq_client import GroqClient

        result = GroqClient._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_markdown_code_block(self):
        """Parses JSON wrapped in markdown code blocks."""
        from finsight_mac.llm.groq_client import GroqClient

        raw = '```json\n{"key": "value"}\n```'
        result = GroqClient._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parse_with_thinking_tags(self):
        """Strips <think>...</think> blocks before parsing."""
        from finsight_mac.llm.groq_client import GroqClient

        raw = '<think>Let me think about this...</think>\n{"result": 42}'
        result = GroqClient._parse_json_response(raw)
        assert result == {"result": 42}

    def test_parse_trailing_commas(self):
        """Handles trailing commas (common LLM mistake)."""
        from finsight_mac.llm.groq_client import GroqClient

        raw = '{"a": 1, "b": 2,}'
        result = GroqClient._parse_json_response(raw)
        assert result == {"a": 1, "b": 2}

    def test_parse_invalid_json_raises(self):
        """Raises JSONDecodeError for unparseable content."""
        from finsight_mac.llm.groq_client import GroqClient

        with pytest.raises(json.JSONDecodeError):
            GroqClient._parse_json_response("not json at all")

    def test_chat_json_sync(self):
        """chat_json() returns parsed dict."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI") as mock_openai, patch(f"{_P}.AsyncOpenAI"):
                from finsight_mac.llm.groq_client import GroqClient

                mock_sync = MagicMock()
                mock_sync.chat.completions.create.return_value = _mock_completion(
                    '{"analysis": "bullish"}'
                )
                mock_openai.return_value = mock_sync

                client = GroqClient()
                result = client.chat_json("Analyze AAPL")
                assert result == {"analysis": "bullish"}

    def test_achat_json_async(self):
        """achat_json() returns parsed dict from async call."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI"), patch(f"{_P}.AsyncOpenAI") as mock_async_openai:
                from finsight_mac.llm.groq_client import GroqClient

                mock_async = MagicMock()
                mock_async.chat.completions.create = AsyncMock(
                    return_value=_mock_completion('{"result": "done"}')
                )
                mock_async_openai.return_value = mock_async

                client = GroqClient()
                result = asyncio.get_event_loop().run_until_complete(
                    client.achat_json("Parse this")
                )
                assert result == {"result": "done"}


class TestGroqHealthCheck:
    """Tests for health check."""

    def test_check_health_success(self):
        """check_health returns True when API responds."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI"), patch(f"{_P}.AsyncOpenAI") as mock_async_openai:
                from finsight_mac.llm.groq_client import GroqClient

                mock_async = MagicMock()
                mock_async.chat.completions.create = AsyncMock(return_value=_mock_completion("ok"))
                mock_async_openai.return_value = mock_async

                client = GroqClient()
                result = asyncio.get_event_loop().run_until_complete(client.check_health())
                assert result is True

    def test_check_health_failure(self):
        """check_health returns False when API is unreachable."""
        with patch(f"{_P}.get_settings", return_value=_mock_settings()):
            with patch(f"{_P}.OpenAI"), patch(f"{_P}.AsyncOpenAI") as mock_async_openai:
                from finsight_mac.llm.groq_client import GroqClient

                mock_async = MagicMock()
                mock_async.chat.completions.create = AsyncMock(
                    side_effect=Exception("Connection refused")
                )
                mock_async_openai.return_value = mock_async

                client = GroqClient()
                result = asyncio.get_event_loop().run_until_complete(client.check_health())
                assert result is False


class TestBuildMessages:
    """Tests for _build_messages helper."""

    def test_user_only(self):
        """Returns just user message when no system message."""
        from finsight_mac.llm.groq_client import GroqClient

        messages = GroqClient._build_messages(None, "hello")
        assert messages == [{"role": "user", "content": "hello"}]

    def test_system_and_user(self):
        """Returns both system and user messages."""
        from finsight_mac.llm.groq_client import GroqClient

        messages = GroqClient._build_messages("be helpful", "hello")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
