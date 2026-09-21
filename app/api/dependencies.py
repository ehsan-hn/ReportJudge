"""FastAPI Dependency Injection Providers.

Decouples HTTP routing and controller logic from domain service instantiation
and LLM client lifecycle management.
"""

from functools import lru_cache

from app.config import settings
from app.judgment.llm.base import LLMClient
from app.judgment.llm.factory import get_llm_client
from app.judgment.service import AssessmentService


@lru_cache
def get_assessment_service() -> AssessmentService:
    """Dependency provider returning an AssessmentService instance.

    Instantiates the configured LLMClient via the factory using application
    settings, and injects it into the domain AssessmentService.
    Cached as a singleton for efficiency across requests.
    """
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "gemini":
        api_key = settings.gemini_api_key
        model = settings.gemini_model
        timeout_seconds = settings.gemini_timeout_seconds
    elif provider == "openai":
        api_key = settings.openai_api_key
        model = settings.openai_model
        timeout_seconds = settings.openai_timeout_seconds
    else:
        api_key = None
        model = None
        timeout_seconds = 30.0

    client: LLMClient = get_llm_client(
        provider=settings.llm_provider,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    return AssessmentService(llm_client=client)
