"""Unit tests for the GeminiLLMClient.

Tests mocked Gemini structured outputs execution including successful parsed completions,
JSON fallback parsing, prompt blocks, candidate safety blocks, empty candidates,
null parsed responses, timeouts, API errors, and unexpected exceptions.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from unittest.mock import AsyncMock, MagicMock
from google.genai import errors, types
import httpx
import pytest

from app.judgment.contracts import (
    DimensionSignal,
    LLMAssessment,
    SeveritySignals,
)
from app.judgment.exceptions import LLMProviderError
from app.judgment.llm.base import LLMClient
from app.judgment.llm.gemini_client import GeminiLLMClient
from app.judgment.rubric import QualityDimension


# ==============================================================================
# Helper Fixtures
# ==============================================================================


def create_dummy_assessment(summary: str = "Test assessment") -> LLMAssessment:
    """Helper to create a valid minimal LLMAssessment fixture."""
    dims = {
        dim: DimensionSignal(
            justification=f"Justification for {dim.value}",
            evidence_quote=None,
            score=3,
            confidence=0.8,
        )
        for dim in QualityDimension
    }
    sev = SeveritySignals(
        reasoning="Test severity reasoning",
        impact_scope=2,
        business_criticality=2,
        time_sensitivity=1,
        confidence=0.85,
    )
    return LLMAssessment(
        is_valid_incident_report=True,
        dimensions=dims,
        severity_signals=sev,
        missing_information=[],
        missing_information_notes="",
        summary=summary,
    )


# ==============================================================================
# GeminiLLMClient Unit Tests
# ==============================================================================


class TestGeminiLLMClient:
    """Unit tests for GeminiLLMClient using mocked google-genai Client."""

    def test_init_sets_properties_and_http_options(self) -> None:
        """Verify initialization configures model, temperature, and timeout options."""
        client = GeminiLLMClient(
            api_key="fake-gemini-key",
            model="gemini-3.8-flash",
            temperature=0.0,
            timeout_seconds=45.0,
        )
        assert isinstance(client, LLMClient)
        assert client.model == "gemini-3.8-flash"
        assert client.temperature == 0.0
        assert client.timeout_seconds == 45.0

    @pytest.mark.asyncio
    async def test_complete_assessment_success_with_parsed(self) -> None:
        """Verify successful call returning response.parsed as an LLMAssessment instance."""
        expected_assessment = create_dummy_assessment(summary="Mocked Gemini assessment")

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.STOP
        mock_candidate.finish_message = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]
        mock_response.parsed = expected_assessment
        mock_response.text = None

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(
            api_key="fake-key",
            model="gemini-3.8-flash",
            client=mock_genai_client,
        )

        system_prompt = "You are an expert incident evaluator."
        user_prompt = "Evaluate this report."

        result = await client.complete_assessment(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        assert result == expected_assessment
        mock_genai_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_genai_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.8-flash"
        assert call_kwargs["contents"] == user_prompt
        config = call_kwargs["config"]
        assert config.system_instruction == system_prompt
        assert config.response_mime_type == "application/json"
        assert config.response_schema == LLMAssessment

    @pytest.mark.asyncio
    async def test_complete_assessment_success_with_parsed_dict(self) -> None:
        """Verify successful call returning response.parsed as a dictionary."""
        expected_assessment = create_dummy_assessment()

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.STOP
        mock_candidate.finish_message = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]
        mock_response.parsed = expected_assessment.model_dump()
        mock_response.text = None

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        result = await client.complete_assessment("sys", "user")

        assert result == expected_assessment

    @pytest.mark.asyncio
    async def test_complete_assessment_fallback_to_text_json(self) -> None:
        """Verify fallback to response.text JSON when response.parsed is None."""
        expected_assessment = create_dummy_assessment(summary="Parsed from JSON text")

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.STOP
        mock_candidate.finish_message = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]
        mock_response.parsed = None
        mock_response.text = expected_assessment.model_dump_json()

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        result = await client.complete_assessment("sys", "user")

        assert result == expected_assessment

    @pytest.mark.asyncio
    async def test_complete_assessment_invalid_text_json_raises_llm_provider_error(self) -> None:
        """Verify invalid response.text JSON raises LLMProviderError."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.STOP
        mock_candidate.finish_message = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]
        mock_response.parsed = None
        mock_response.text = "invalid json {not valid}"

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(LLMProviderError, match="Failed to validate assessment from response text"):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_prompt_blocked_raises_llm_provider_error(self) -> None:
        """Verify prompt-level block in prompt_feedback raises LLMProviderError."""
        mock_feedback = MagicMock()
        mock_feedback.block_reason = "SAFETY"

        mock_response = MagicMock()
        mock_response.prompt_feedback = mock_feedback
        mock_response.candidates = []

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(LLMProviderError, match="Prompt was blocked by Gemini: SAFETY"):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_empty_candidates_raises_llm_provider_error(self) -> None:
        """Verify empty candidates list raises LLMProviderError."""
        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = []

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(LLMProviderError, match="Gemini returned no candidates."):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_safety_finish_reason_raises_llm_provider_error(self) -> None:
        """Verify safety finish reason raises LLMProviderError with refusal details."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.SAFETY
        mock_candidate.finish_message = "Content violated safety policies."

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(
            LLMProviderError,
            match="Model refused or blocked assessment: Content violated safety policies.",
        ):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_recitation_finish_reason_raises_llm_provider_error(self) -> None:
        """Verify recitation finish reason raises LLMProviderError."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.RECITATION
        mock_candidate.finish_message = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(
            LLMProviderError,
            match="Model refused or blocked assessment: RECITATION",
        ):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_null_parsed_and_empty_text_raises_llm_provider_error(self) -> None:
        """Verify null parsed and empty text raises LLMProviderError."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.STOP
        mock_candidate.finish_message = None

        mock_response = MagicMock()
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]
        mock_response.parsed = None
        mock_response.text = ""

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(
            LLMProviderError,
            match="Gemini returned null parsed assessment output.",
        ):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_api_error_wrapped(self) -> None:
        """Verify google.genai.errors.APIError is caught and wrapped into LLMProviderError."""
        api_error = errors.APIError(code=429, response_json={"error": {"message": "Quota exceeded for model"}})

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(side_effect=api_error)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(LLMProviderError) as exc_info:
            await client.complete_assessment("sys", "user")

        assert "Quota exceeded" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_error

    @pytest.mark.asyncio
    async def test_complete_assessment_timeout_wrapped(self) -> None:
        """Verify httpx.TimeoutException is caught and wrapped into LLMProviderError."""
        timeout_err = httpx.TimeoutException("Read timeout after 30s")

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(side_effect=timeout_err)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(LLMProviderError) as exc_info:
            await client.complete_assessment("sys", "user")

        assert "Gemini request timed out" in str(exc_info.value)
        assert exc_info.value.__cause__ is timeout_err

    @pytest.mark.asyncio
    async def test_complete_assessment_unexpected_exception_wrapped(self) -> None:
        """Verify unexpected exceptions are caught and wrapped into LLMProviderError."""
        unexpected_err = ConnectionResetError("Connection abruptly closed by peer")

        mock_genai_client = MagicMock()
        mock_genai_client.aio.models.generate_content = AsyncMock(side_effect=unexpected_err)

        client = GeminiLLMClient(api_key="fake-key", client=mock_genai_client)
        with pytest.raises(LLMProviderError) as exc_info:
            await client.complete_assessment("sys", "user")

        assert "Connection abruptly closed by peer" in str(exc_info.value)
        assert exc_info.value.__cause__ is unexpected_err
