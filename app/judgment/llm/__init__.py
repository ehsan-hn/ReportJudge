"""LLM Client interfaces and implementations for incident judgment.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from .base import LLMClient
from .fake_client import FakeLLMClient

__all__ = ["LLMClient", "FakeLLMClient"]
