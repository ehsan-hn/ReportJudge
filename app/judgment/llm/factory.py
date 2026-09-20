"""LLM Client Factory.

Provides factory function to instantiate concrete LLMClient implementations based on provider configuration.
Supported providers:
- "fake": Offline deterministic mock client with optional scripted responses or heuristic fallback.
- "openai": Live OpenAI Structured Outputs client using AsyncOpenAI.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from typing import Any

from .base import LLMClient
from .fake_client import FakeLLMClient
from .openai_client import OpenAILLMClient


def get_llm_client(
    provider: str = "fake",
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    **kwargs: Any,
) -> LLMClient:
    """Instantiate and return an LLMClient instance according to provider.

    Args:
        provider: Provider identifier ("fake" or "openai"). Case-insensitive.
        api_key: OpenAI API key (required when provider is "openai").
        model: Model identifier (defaults to "gpt-4o-mini").
        **kwargs: Additional parameters passed to the client constructor
            (e.g., scripted_responses for FakeLLMClient, temperature or timeout_seconds for OpenAILLMClient).

    Returns:
        LLMClient: The initialized client instance.

    Raises:
        ValueError: If api_key is missing when provider is "openai", or if provider is unsupported.
    """
    provider_clean = (provider or "").strip().lower()

    if provider_clean == "fake":
        return FakeLLMClient(**kwargs)

    if provider_clean == "openai":
        if not api_key or not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
        return OpenAILLMClient(api_key=api_key, model=model, **kwargs)

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. Supported: ['fake', 'openai']"
    )
