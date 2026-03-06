"""Tests for mac application configuration."""

from __future__ import annotations


class TestConfig:
    def test_settings_defaults(self) -> None:
        from finsight_mac.config import Settings

        settings = Settings(
            ollama_model="qwen3.5:9b",
            ollama_base_url="http://localhost:11434",
        )
        assert settings.ollama_model == "qwen3.5:9b"
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_settings_openai_url(self) -> None:
        from finsight_mac.config import Settings

        settings = Settings(
            ollama_model="qwen3:8b",
            ollama_base_url="http://localhost:11434",
        )
        expected = "http://localhost:11434/v1"
        assert settings.ollama_openai_url == expected
