"""LLM Client interfaces and implementations for incident judgment.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from .base import LLMClient
from .factory import get_llm_client
from .fake_client import FakeLLMClient
from .openai_client import OpenAILLMClient

__all__ = ["LLMClient", "FakeLLMClient", "OpenAILLMClient", "get_llm_client"]
