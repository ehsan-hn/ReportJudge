"""LLM Client Factory.

Provides factory function to instantiate concrete LLMClient implementations based on provider configuration.
Supported providers:
- "fake": Offline deterministic mock client with optional scripted responses or heuristic fallback.
- "openai": Live OpenAI Structured Outputs client using AsyncOpenAI.
- "gemini": Live Google Gemini Structured Outputs client using google-genai.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from typing import Any

from .base import LLMClient
from .fake_client import FakeLLMClient
from .gemini_client import GeminiLLMClient
from .openai_client import OpenAILLMClient


def get_llm_client(
    provider: str = "fake",
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMClient:
    """Instantiate and return an LLMClient instance according to provider.

    Args:
        provider: Provider identifier ("fake", "openai", or "gemini"). Case-insensitive.
        api_key: API key (required when provider is "openai" or "gemini").
        model: Model identifier (defaults to "gpt-4o-mini" for OpenAI, "gemini-3.8-flash" for Gemini).
        **kwargs: Additional parameters passed to the client constructor
            (e.g., scripted_responses for FakeLLMClient, temperature or timeout_seconds for live clients).

    Returns:
        LLMClient: The initialized client instance.

    Raises:
        ValueError: If api_key is missing when provider is "openai" or "gemini", or if provider is unsupported.
    """
    provider_clean = (provider or "").strip().lower()

    if provider_clean == "fake":
        return FakeLLMClient(**kwargs)

    if provider_clean == "openai":
        if not api_key or not api_key.strip():
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")
        openai_model = model or "gpt-4o-mini"
        return OpenAILLMClient(api_key=api_key, model=openai_model, **kwargs)

    if provider_clean == "gemini":
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'")
        gemini_model = "gemini-3.8-flash" if (model is None or model == "gpt-4o-mini") else model
        return GeminiLLMClient(api_key=api_key, model=gemini_model, **kwargs)

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. Supported: ['fake', 'openai', 'gemini']"
    )
