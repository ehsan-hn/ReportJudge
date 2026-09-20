"""Live OpenAI Structured Outputs Client.

Uses native OpenAI Structured Outputs (client.beta.chat.completions.parse with
response_format=LLMAssessment) to enforce strict schema-compliant assessments.
All network, timeout, API, and refusal errors are mapped to the domain exception LLMProviderError.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from typing import Any
import openai
from openai import AsyncOpenAI

from ..contracts import LLMAssessment
from ..exceptions import LLMProviderError
from .base import LLMClient


class OpenAILLMClient(LLMClient):
    """OpenAI Structured Outputs client implementation using AsyncOpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAILLMClient.

        Args:
            api_key: The OpenAI API key.
            model: Model identifier (defaults to "gpt-4o-mini").
            temperature: Sampling temperature (defaults to 0.0 for deterministic evaluation).
            timeout_seconds: Request timeout in seconds.
            max_retries: Maximum retry attempts for transient failures.
            client: Optional pre-configured AsyncOpenAI client instance (useful for testing/mocking).
            **kwargs: Extra arguments forwarded to AsyncOpenAI constructor if client is not provided.
        """
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = client or AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
            **kwargs,
        )

    async def complete_assessment(
        self, system_prompt: str, user_prompt: str
    ) -> LLMAssessment:
        """Completes semantic assessment of the incident report.

        Args:
            system_prompt: System prompt instructing the model on rubric and formatting.
            user_prompt: User prompt containing the formatted/sanitized report.

        Returns:
            LLMAssessment: Fully validated structured assessment output.

        Raises:
            LLMProviderError: If the model refuses or an API / network / unexpected error occurs.
        """
        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=LLMAssessment,
            )

            if not response.choices:
                raise LLMProviderError("OpenAI returned an empty choices list.")

            choice = response.choices[0]
            if choice.message.refusal:
                raise LLMProviderError(
                    f"Model refused assessment: {choice.message.refusal}"
                )

            if choice.message.parsed is None:
                raise LLMProviderError(
                    "OpenAI returned null parsed assessment output."
                )

            return choice.message.parsed
        except LLMProviderError:
            raise
        except (openai.APITimeoutError, openai.RateLimitError, openai.APIError) as exc:
            raise LLMProviderError(str(exc)) from exc
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc
