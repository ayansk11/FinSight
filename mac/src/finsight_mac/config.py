"""FinSight Mac configuration.

Settings are loaded from environment variables (.env file) with sensible defaults.
Model selection is config-driven — users can switch between Qwen3.5-9B and Qwen3-8B.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- LLM Configuration ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_model: str = Field(
        default="qwen3.5:9b",
        description=(
            "Ollama model name. Options: qwen3.5:9b (default), "
            "qwen3:8b-q4_K_M (RLM), qwen3.5:4b, qwen3.5:2b, qwen3.5:0.8b"
        ),
    )
    ollama_timeout: int = Field(
        default=300,
        description="Request timeout in seconds for Ollama API calls",
    )
    max_context_tokens: int = Field(
        default=32000,
        description="Maximum context window size in tokens",
    )
    temperature: float = Field(
        default=0.1,
        description="LLM temperature (low for factual analysis)",
    )

    # --- PageIndex Configuration ---
    pageindex_api_key: str | None = Field(
        default=None,
        description="PageIndex Cloud API key (free tier: 200 pages)",
    )
    pageindex_model: str = Field(
        default="gpt-4o-2024-11-20",
        description="Model used for PageIndex tree generation (requires OpenAI API key)",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for PageIndex tree generation",
    )

    # --- Data Paths ---
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root data directory",
    )
    filings_dir: Path = Field(
        default=Path("./data/filings"),
        description="Directory for downloaded SEC filings",
    )
    trees_dir: Path = Field(
        default=Path("./data/trees"),
        description="Directory for generated PageIndex trees",
    )

    # --- API Keys (free tiers) ---
    fred_api_key: str | None = Field(
        default=None,
        description="FRED API key for macroeconomic data",
    )
    finnhub_api_key: str | None = Field(
        default=None,
        description="Finnhub API key for market data",
    )
    fmp_api_key: str | None = Field(
        default=None,
        description="Financial Modeling Prep API key",
    )
    alpha_vantage_api_key: str | None = Field(
        default=None,
        description="Alpha Vantage API key for historical data and technical indicators",
    )
    stockdata_api_key: str | None = Field(
        default=None,
        description="StockData.org API key for market news and sentiment",
    )
    sec_user_agent: str = Field(
        default="FinSight research@example.com",
        description="User-Agent header for SEC EDGAR requests",
    )

    # --- Groq Cloud Fallback ---
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key for cloud LLM fallback when Ollama times out",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Groq model name. Options: llama-3.3-70b-versatile (default, best for JSON), "
            "meta-llama/llama-4-scout-17b-16e-instruct, llama-3.1-8b-instant"
        ),
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible API base URL",
    )

    # --- Agent Configuration ---
    max_agent_iterations: int = Field(
        default=20,
        description="Circuit breaker: max iterations per agent",
    )
    max_rlm_iterations: int = Field(
        default=15,
        description="Max RLM REPL iterations for cross-doc queries",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.filings_dir.mkdir(parents=True, exist_ok=True)
        self.trees_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ollama_openai_url(self) -> str:
        """Ollama's OpenAI-compatible API endpoint."""
        return f"{self.ollama_base_url}/v1"


# Global singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
