"""Unit tests for the LLM Provider Factory and OpenAILLMClient.

Tests factory instantiation for "fake" and "openai" providers, validation of
required API keys, handling of unsupported providers, and mocked OpenAI
Structured Outputs execution (success, refusal, rate limits, timeouts, and API errors).

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from unittest.mock import AsyncMock, MagicMock
import httpx
import openai
import pytest

from app.judgment.contracts import (
    DimensionSignal,
    LLMAssessment,
    SeveritySignals,
)
from app.judgment.exceptions import LLMProviderError
from app.judgment.llm.base import LLMClient
from app.judgment.llm.factory import get_llm_client
from app.judgment.llm.fake_client import FakeLLMClient
from app.judgment.llm.gemini_client import GeminiLLMClient
from app.judgment.llm.openai_client import OpenAILLMClient
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
# Factory Tests
# ==============================================================================


class TestLLMFactory:
    """Tests for the get_llm_client factory function."""

    def test_factory_returns_fake_client_default(self) -> None:
        """Calling get_llm_client without arguments returns a FakeLLMClient."""
        client = get_llm_client()
        assert isinstance(client, FakeLLMClient)
        assert isinstance(client, LLMClient)
        assert client.remaining_scripted_responses == 0

    def test_factory_returns_fake_client_explicit(self) -> None:
        """Calling get_llm_client with provider='fake' returns a FakeLLMClient."""
        client = get_llm_client(provider="fake")
        assert isinstance(client, FakeLLMClient)

    def test_factory_fake_client_case_insensitive(self) -> None:
        """Factory handles provider casing insensitively ('FAKE', 'Fake', ' fake ')."""
        for p in ["FAKE", "Fake", "  fake  "]:
            client = get_llm_client(provider=p)
            assert isinstance(client, FakeLLMClient)

    def test_factory_fake_client_passes_scripted_responses(self) -> None:
        """Factory forwards kwargs such as scripted_responses to FakeLLMClient."""
        dummy = create_dummy_assessment()
        client = get_llm_client(provider="fake", scripted_responses=[dummy])
        assert isinstance(client, FakeLLMClient)
        assert client.remaining_scripted_responses == 1

    def test_factory_openai_missing_api_key_raises_value_error(self) -> None:
        """Factory raises ValueError if provider is 'openai' but api_key is None or empty."""
        for invalid_key in [None, "", "   "]:
            with pytest.raises(
                ValueError,
                match="OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'",
            ):
                get_llm_client(provider="openai", api_key=invalid_key)

    def test_factory_openai_missing_api_key_case_insensitive(self) -> None:
        """Factory validates missing api_key regardless of provider casing."""
        with pytest.raises(
            ValueError,
            match="OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'",
        ):
            get_llm_client(provider="OPENAI", api_key=None)

    def test_factory_unsupported_provider_raises_value_error(self) -> None:
        """Factory raises ValueError for unknown provider."""
        with pytest.raises(
            ValueError,
            match=r"Unsupported LLM provider: 'anthropic_unsupported'\. Supported: \['fake', 'openai', 'gemini'\]",
        ):
            get_llm_client(provider="anthropic_unsupported")

    def test_factory_gemini_missing_api_key_raises_value_error(self) -> None:
        """Factory raises ValueError if provider is 'gemini' but api_key is None or empty."""
        for invalid_key in [None, "", "   "]:
            with pytest.raises(
                ValueError,
                match="GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'",
            ):
                get_llm_client(provider="gemini", api_key=invalid_key)

    def test_factory_gemini_missing_api_key_case_insensitive(self) -> None:
        """Factory validates missing api_key regardless of gemini provider casing."""
        with pytest.raises(
            ValueError,
            match="GEMINI_API_KEY is required when LLM_PROVIDER is 'gemini'",
        ):
            get_llm_client(provider="GEMINI", api_key=None)

    def test_factory_gemini_success(self) -> None:
        """Factory creates GeminiLLMClient with provided arguments."""
        client = get_llm_client(
            provider="gemini",
            api_key="gemini-test-key",
            model="gemini-3.8-flash",
            temperature=0.2,
            timeout_seconds=40.0,
        )
        assert isinstance(client, GeminiLLMClient)
        assert isinstance(client, LLMClient)
        assert client.model == "gemini-3.8-flash"
        assert client.temperature == 0.2
        assert client.timeout_seconds == 40.0

    def test_factory_gemini_default_model(self) -> None:
        """Factory defaults to 'gemini-3.8-flash' if model is not provided or defaulted."""
        client = get_llm_client(provider="gemini", api_key="gemini-test-key")
        assert isinstance(client, GeminiLLMClient)
        assert client.model == "gemini-3.8-flash"

    def test_factory_gemini_case_insensitive(self) -> None:
        """Factory creates GeminiLLMClient for upper/mixed case provider string."""
        client = get_llm_client(provider="Gemini", api_key="gemini-test-key")
        assert isinstance(client, GeminiLLMClient)

    def test_factory_openai_success(self) -> None:
        """Factory creates OpenAILLMClient with provided arguments."""
        client = get_llm_client(
            provider="openai",
            api_key="sk-test-factory-key",
            model="gpt-4o-mini",
            temperature=0.1,
            timeout_seconds=45.0,
            max_retries=3,
        )
        assert isinstance(client, OpenAILLMClient)
        assert isinstance(client, LLMClient)
        assert client.model == "gpt-4o-mini"
        assert client.temperature == 0.1
        assert client.timeout_seconds == 45.0
        assert client.max_retries == 3

    def test_factory_openai_case_insensitive(self) -> None:
        """Factory creates OpenAILLMClient for upper/mixed case provider string."""
        client = get_llm_client(provider="OpenAI", api_key="sk-test-key")
        assert isinstance(client, OpenAILLMClient)


# ==============================================================================
# OpenAILLMClient Tests
# ==============================================================================


class TestOpenAILLMClient:
    """Unit tests for OpenAILLMClient using mocked AsyncOpenAI."""

    @pytest.mark.asyncio
    async def test_complete_assessment_success(self) -> None:
        """Verify successful call to complete_assessment returns parsed LLMAssessment."""
        expected_assessment = create_dummy_assessment(summary="Mocked live assessment")

        mock_choice = MagicMock()
        mock_choice.message.refusal = None
        mock_choice.message.parsed = expected_assessment

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client = OpenAILLMClient(
            api_key="sk-test-key",
            model="gpt-4o-mini",
            temperature=0.0,
            timeout_seconds=30.0,
            max_retries=2,
        )
        client.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        system_prompt = "You are an expert incident evaluator."
        user_prompt = "Evaluate this report."

        result = await client.complete_assessment(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        assert result == expected_assessment
        client.client.beta.chat.completions.parse.assert_awaited_once_with(
            model="gpt-4o-mini",
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=LLMAssessment,
        )

    @pytest.mark.asyncio
    async def test_complete_assessment_refusal_raises_llm_provider_error(self) -> None:
        """Verify model refusal raises LLMProviderError with refusal message."""
        mock_choice = MagicMock()
        mock_choice.message.refusal = "Assessment violates content policy."
        mock_choice.message.parsed = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client = OpenAILLMClient(api_key="sk-test-key")
        client.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        with pytest.raises(
            LLMProviderError,
            match="Model refused assessment: Assessment violates content policy.",
        ):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_empty_choices_raises_llm_provider_error(self) -> None:
        """Verify empty choices returned raises LLMProviderError."""
        mock_response = MagicMock()
        mock_response.choices = []

        client = OpenAILLMClient(api_key="sk-test-key")
        client.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        with pytest.raises(
            LLMProviderError,
            match="OpenAI returned an empty choices list.",
        ):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_null_parsed_raises_llm_provider_error(self) -> None:
        """Verify null parsed output without explicit refusal raises LLMProviderError."""
        mock_choice = MagicMock()
        mock_choice.message.refusal = None
        mock_choice.message.parsed = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client = OpenAILLMClient(api_key="sk-test-key")
        client.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        with pytest.raises(
            LLMProviderError,
            match="OpenAI returned null parsed assessment output.",
        ):
            await client.complete_assessment("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_assessment_rate_limit_error_wrapped(self) -> None:
        """Verify openai.RateLimitError is caught and wrapped into LLMProviderError."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(429, request=req)
        rate_limit_err = openai.RateLimitError("Rate limit exceeded", response=resp, body=None)

        client = OpenAILLMClient(api_key="sk-test-key")
        client.client.beta.chat.completions.parse = AsyncMock(side_effect=rate_limit_err)

        with pytest.raises(LLMProviderError) as exc_info:
            await client.complete_assessment("sys", "user")

        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.__cause__ is rate_limit_err

    @pytest.mark.asyncio
    async def test_complete_assessment_timeout_error_wrapped(self) -> None:
        """Verify openai.APITimeoutError is caught and wrapped into LLMProviderError."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        timeout_err = openai.APITimeoutError(request=req)

        client = OpenAILLMClient(api_key="sk-test-key")
        client.client.beta.chat.completions.parse = AsyncMock(side_effect=timeout_err)

        with pytest.raises(LLMProviderError) as exc_info:
            await client.complete_assessment("sys", "user")

        assert "Request timed out" in str(exc_info.value)
        assert exc_info.value.__cause__ is timeout_err

    @pytest.mark.asyncio
    async def test_complete_assessment_api_error_wrapped(self) -> None:
        """Verify general openai.APIError is caught and wrapped into LLMProviderError."""
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        api_err = openai.APIError("Service temporarily unavailable", request=req, body=None)

        client = OpenAILLMClient(api_key="sk-test-key")
        client.client.beta.chat.completions.parse = AsyncMock(side_effect=api_err)

        with pytest.raises(LLMProviderError) as exc_info:
            await client.complete_assessment("sys", "user")

        assert "Service temporarily unavailable" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_err

    @pytest.mark.asyncio
    async def test_complete_assessment_unexpected_exception_wrapped(self) -> None:
        """Verify unexpected exceptions are caught and wrapped into LLMProviderError."""
        unexpected_err = ConnectionResetError("Connection abruptly closed by peer")

        client = OpenAILLMClient(api_key="sk-test-key")
        client.client.beta.chat.completions.parse = AsyncMock(side_effect=unexpected_err)

        with pytest.raises(LLMProviderError) as exc_info:
            await client.complete_assessment("sys", "user")

        assert "Connection abruptly closed by peer" in str(exc_info.value)
        assert exc_info.value.__cause__ is unexpected_err
