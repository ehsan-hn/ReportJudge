"""Live Gemini Structured Outputs Client.

Uses native Google GenAI SDK (client.aio.models.generate_content with
response_schema=LLMAssessment and response_mime_type="application/json")
to enforce strict schema-compliant assessments.
All network, timeout, API, refusal, and block errors are mapped to the domain exception LLMProviderError.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

import json
from typing import Any
from google import genai
from google.genai import errors, types
import httpx
from pydantic import ValidationError

from ..contracts import LLMAssessment
from ..exceptions import LLMProviderError
from .base import LLMClient


class GeminiLLMClient(LLMClient):
    """Google Gemini Structured Outputs client implementation using google-genai."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        client: genai.Client | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the GeminiLLMClient.

        Args:
            api_key: The Google Gemini API key.
            model: Model identifier (defaults to "gemini-3.8-flash").
            temperature: Sampling temperature (defaults to 0.0 for deterministic evaluation).
            timeout_seconds: Request timeout in seconds.
            client: Optional pre-configured genai.Client instance (useful for testing/mocking).
            **kwargs: Extra arguments forwarded to genai.Client constructor if client is not provided.
        """
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        timeout_ms = int(timeout_seconds * 1000)

        http_options = kwargs.pop("http_options", None)
        if http_options is None:
            http_options = types.HttpOptions(timeout=timeout_ms)
        elif isinstance(http_options, dict) and "timeout" not in http_options:
            http_options["timeout"] = timeout_ms

        self.client = client or genai.Client(
            api_key=api_key,
            http_options=http_options,
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
            LLMProviderError: If the model refuses, prompt is blocked, or an API / network / unexpected error occurs.
        """
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=LLMAssessment,
            )

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )

            # Check for prompt-level block
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise LLMProviderError(
                    f"Prompt was blocked by Gemini: {response.prompt_feedback.block_reason}"
                )

            # Check candidates presence
            if not response.candidates:
                raise LLMProviderError("Gemini returned no candidates.")

            candidate = response.candidates[0]

            # Check for safety / block / refusal finish reasons
            if candidate.finish_reason is not None:
                finish_reason_str = str(
                    candidate.finish_reason.value
                    if hasattr(candidate.finish_reason, "value")
                    else candidate.finish_reason
                ).upper()
                blocked_names = {
                    "SAFETY",
                    "RECITATION",
                    "BLOCKLIST",
                    "PROHIBITED_CONTENT",
                    "SPII",
                    "IMAGE_SAFETY",
                }
                if finish_reason_str in blocked_names:
                    finish_msg = getattr(candidate, "finish_message", None) or finish_reason_str
                    raise LLMProviderError(
                        f"Model refused or blocked assessment: {finish_msg}"
                    )

            # Check parsed output
            if response.parsed is not None:
                if isinstance(response.parsed, LLMAssessment):
                    return response.parsed
                if isinstance(response.parsed, dict):
                    return LLMAssessment.model_validate(response.parsed)
                return response.parsed

            # Fallback to response.text if parsed is None
            if response.text and response.text.strip():
                try:
                    return LLMAssessment.model_validate_json(response.text)
                except (ValidationError, json.JSONDecodeError) as exc:
                    raise LLMProviderError(
                        f"Failed to validate assessment from response text: {exc}"
                    ) from exc

            raise LLMProviderError(
                "Gemini returned null parsed assessment output."
            )

        except LLMProviderError:
            raise
        except errors.APIError as exc:
            raise LLMProviderError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"Gemini request timed out: {exc}") from exc
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc
