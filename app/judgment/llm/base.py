"""Abstract LLM Client Interface.

Defines the abstract interface for LLM completions in the incident judgment domain.
Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from abc import ABC, abstractmethod

from ..contracts import LLMAssessment


class LLMClient(ABC):
    """Abstract interface for LLM completions evaluating incident reports."""

    @abstractmethod
    async def complete_assessment(
        self, system_prompt: str, user_prompt: str
    ) -> LLMAssessment:
        """Completes semantic assessment of the incident report.

        Args:
            system_prompt: System prompt instructing the model on rubric and formatting.
            user_prompt: User prompt containing the formatted/sanitized report.

        Returns:
            LLMAssessment: Fully validated structured assessment output.
        """
        pass
