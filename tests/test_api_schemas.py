"""Tests for API DTO Schemas and Application Settings.

Verifies strict input validation boundaries, length bounds, extra field prohibition,
JSON serialization, domain model compatibility, and settings configuration.
"""

from uuid import UUID, uuid4
import pytest
from pydantic import ValidationError

from app.api.v1.schemas import (
    AssessmentOut,
    ConfidenceBreakdown,
    DimensionDetail,
    IncidentInput,
    QualityBreakdown,
    SeverityBreakdown,
)
from app.config import Settings, settings
from app.judgment.rubric import MissingInfoCode, SeverityLevel
from app.judgment.service import AssessmentService


# ==============================================================================
# 1. IncidentInput Request Schema Tests
# ==============================================================================


def test_incident_input_valid_payload_minimal() -> None:
    """Verify IncidentInput parses minimal valid payload with only required fields."""
    payload = {
        "title": "Checkout API returning 500",
        "description": "Users cannot complete purchases on payment step.",
    }
    incident = IncidentInput(**payload)
    assert incident.title == "Checkout API returning 500"
    assert incident.description == "Users cannot complete purchases on payment step."
    assert incident.impact is None
    assert incident.evidence is None
    assert incident.actions_taken is None


def test_incident_input_valid_payload_full() -> None:
    """Verify IncidentInput parses complete payload with all optional fields."""
    payload = {
        "title": "Checkout API returning 500",
        "description": "Users cannot complete purchases on payment step.",
        "impact": "Roughly 20% of European traffic failing.",
        "evidence": "Datadog alert #502, 5xx rate spiked to 18%.",
        "actions_taken": "Scaled pods to 12; no resolution.",
    }
    incident = IncidentInput(**payload)
    assert incident.title == payload["title"]
    assert incident.description == payload["description"]
    assert incident.impact == payload["impact"]
    assert incident.evidence == payload["evidence"]
    assert incident.actions_taken == payload["actions_taken"]


def test_incident_input_rejects_missing_title() -> None:
    """Verify IncidentInput rejects payloads missing required 'title' field."""
    payload = {
        "description": "Users cannot complete purchases on payment step.",
    }
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(**payload)
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("title",) and err["type"] == "missing" for err in errors)


def test_incident_input_rejects_missing_description() -> None:
    """Verify IncidentInput rejects payloads missing required 'description' field."""
    payload = {
        "title": "Checkout API returning 500",
    }
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(**payload)
    errors = exc_info.value.errors()
    assert any(err["loc"] == ("description",) and err["type"] == "missing" for err in errors)


def test_incident_input_title_length_bounds() -> None:
    """Verify IncidentInput enforces title min_length=3 and max_length=200."""
    # Under min length: 2 characters
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(title="ab", description="Valid description")
    assert any("title" in str(err["loc"]) for err in exc_info.value.errors())

    # Whitespace only under min length
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(title="   ", description="Valid description")
    assert any("title" in str(err["loc"]) for err in exc_info.value.errors())

    # Exactly min length: 3 characters
    incident_min = IncidentInput(title="SEV", description="Valid description")
    assert incident_min.title == "SEV"

    # Exactly max length: 200 characters
    title_200 = "T" * 200
    incident_max = IncidentInput(title=title_200, description="Valid description")
    assert len(incident_max.title) == 200

    # Over max length: 201 characters
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(title="T" * 201, description="Valid description")
    assert any("title" in str(err["loc"]) for err in exc_info.value.errors())


def test_incident_input_description_length_bounds() -> None:
    """Verify IncidentInput enforces description min_length=5 and max_length=8000."""
    # Under min length: 4 characters
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(title="Valid Title", description="Fail")
    assert any("description" in str(err["loc"]) for err in exc_info.value.errors())

    # Exactly min length: 5 characters
    incident_min = IncidentInput(title="Valid Title", description="12345")
    assert incident_min.description == "12345"

    # Exactly max length: 8000 characters
    desc_8000 = "D" * 8000
    incident_max = IncidentInput(title="Valid Title", description=desc_8000)
    assert len(incident_max.description) == 8000

    # Over max length: 8001 characters
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(title="Valid Title", description="D" * 8001)
    assert any("description" in str(err["loc"]) for err in exc_info.value.errors())


def test_incident_input_optional_fields_length_bounds() -> None:
    """Verify length bounds for optional impact, evidence, and actions_taken."""
    # Impact over 2000 chars
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(
            title="Valid Title",
            description="Valid description",
            impact="I" * 2001,
        )
    assert any("impact" in str(err["loc"]) for err in exc_info.value.errors())

    # Evidence over 4000 chars
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(
            title="Valid Title",
            description="Valid description",
            evidence="E" * 4001,
        )
    assert any("evidence" in str(err["loc"]) for err in exc_info.value.errors())

    # Actions taken over 2000 chars
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(
            title="Valid Title",
            description="Valid description",
            actions_taken="A" * 2001,
        )
    assert any("actions_taken" in str(err["loc"]) for err in exc_info.value.errors())


def test_incident_input_rejects_extra_unrecognized_attributes() -> None:
    """Verify extra='forbid' strictly rejects unknown input fields at the gateway."""
    payload = {
        "title": "Valid Title",
        "description": "Valid incident description.",
        "extra_field": "hacker_payload",
        "nested_unrecognized": {"foo": "bar"},
    }
    with pytest.raises(ValidationError) as exc_info:
        IncidentInput(**payload)
    errors = exc_info.value.errors()
    forbidden_locs = {err["loc"][0] for err in errors if err["type"] == "extra_forbidden"}
    assert "extra_field" in forbidden_locs
    assert "nested_unrecognized" in forbidden_locs


# ==============================================================================
# 2. Response Sub-Schemas Validation Tests
# ==============================================================================


def test_dimension_detail_validation() -> None:
    """Verify DimensionDetail bounds and score constraint ge=0, le=5."""
    dim = DimensionDetail(
        dimension="CLARITY",
        score=4,
        weight=0.20,
        justification="Very clear explanation",
        confidence=0.9,
    )
    assert dim.score == 4

    # Score below 0
    with pytest.raises(ValidationError):
        DimensionDetail(
            dimension="CLARITY",
            score=-1,
            weight=0.20,
            justification="Negative score",
            confidence=0.9,
        )

    # Score above 5
    with pytest.raises(ValidationError):
        DimensionDetail(
            dimension="CLARITY",
            score=6,
            weight=0.20,
            justification="Too high score",
            confidence=0.9,
        )


def test_quality_breakdown_validation() -> None:
    """Verify QualityBreakdown score constraint ge=0.0, le=100.0."""
    detail = DimensionDetail(
        dimension="CLARITY",
        score=4,
        weight=0.20,
        justification="Clear",
        confidence=0.9,
    )
    qb = QualityBreakdown(score=85.0, band="GOOD", dimensions=[detail])
    assert qb.score == 85.0

    # Score below 0.0
    with pytest.raises(ValidationError):
        QualityBreakdown(score=-0.1, band="POOR", dimensions=[detail])

    # Score above 100.0
    with pytest.raises(ValidationError):
        QualityBreakdown(score=100.1, band="EXCELLENT", dimensions=[detail])


def test_severity_breakdown_validation() -> None:
    """Verify SeverityBreakdown severity level enum and rules trace."""
    sb = SeverityBreakdown(
        level=SeverityLevel.SEV2,
        index=2.1,
        applied_rules=["TOTAL_OUTAGE_SEV2"],
    )
    assert sb.level == SeverityLevel.SEV2
    assert sb.applied_rules == ["TOTAL_OUTAGE_SEV2"]

    # String coercion to enum
    sb_str = SeverityBreakdown(level="SEV1", index=1.0)
    assert sb_str.level == SeverityLevel.SEV1

    # Invalid severity level
    with pytest.raises(ValidationError):
        SeverityBreakdown(level="SEV5", index=5.0)


def test_confidence_breakdown_validation() -> None:
    """Verify ConfidenceBreakdown bounds constraint ge=0.0, le=1.0."""
    cb = ConfidenceBreakdown(
        value=0.88,
        band="HIGH",
        limiting_reason=None,
        factors={"model_self_reported": 0.9},
    )
    assert cb.value == 0.88

    # Value below 0.0
    with pytest.raises(ValidationError):
        ConfidenceBreakdown(value=-0.01, band="LOW")

    # Value above 1.0
    with pytest.raises(ValidationError):
        ConfidenceBreakdown(value=1.01, band="HIGH")


# ==============================================================================
# 3. AssessmentOut Serialization & Integration Tests
# ==============================================================================


def test_assessment_out_serializes_to_valid_json() -> None:
    """Verify AssessmentOut serializes cleanly to JSON and validates types."""
    custom_uuid = uuid4()
    assessment = AssessmentOut(
        assessment_id=custom_uuid,
        status="assessed",
        report_quality=QualityBreakdown(
            score=82.5,
            band="GOOD",
            dimensions=[
                DimensionDetail(
                    dimension="CLARITY",
                    score=4,
                    weight=0.20,
                    justification="Clear description",
                    confidence=0.9,
                )
            ],
        ),
        severity=SeverityBreakdown(
            level=SeverityLevel.SEV2,
            index=2.1,
            applied_rules=["TOTAL_OUTAGE_SEV2"],
        ),
        confidence=ConfidenceBreakdown(
            value=0.85,
            band="HIGH",
            limiting_reason=None,
            factors={"model_self_reported": 0.9, "coverage_factor": 1.0},
        ),
        missing_information=[MissingInfoCode.NO_REPRO_STEPS],
        missing_information_notes="Repro steps omitted",
        summary_explanation="Overall critical incident.",
        requires_human_review=False,
    )

    # Check types and properties
    assert isinstance(assessment.assessment_id, UUID)
    assert assessment.assessment_id == custom_uuid
    assert assessment.severity.level == SeverityLevel.SEV2
    assert assessment.missing_information == [MissingInfoCode.NO_REPRO_STEPS]

    # JSON serialization roundtrip
    json_str = assessment.model_dump_json()
    assert isinstance(json_str, str)
    assert str(custom_uuid) in json_str
    assert "NO_REPRO_STEPS" in json_str
    assert "SEV2" in json_str

    # JSON deserialization roundtrip
    reloaded = AssessmentOut.model_validate_json(json_str)
    assert reloaded.assessment_id == custom_uuid
    assert reloaded.status == "assessed"
    assert reloaded.report_quality.score == 82.5
    assert reloaded.severity.level == SeverityLevel.SEV2
    assert reloaded.missing_information == [MissingInfoCode.NO_REPRO_STEPS]


def test_assessment_out_generates_default_uuid() -> None:
    """Verify AssessmentOut generates fresh UUID4 by default when omitted."""
    data = {
        "status": "not_an_incident_report",
        "report_quality": {
            "score": 0.0,
            "band": "POOR",
            "dimensions": [],
        },
        "severity": {
            "level": "SEV4",
            "index": 0.0,
            "applied_rules": [],
        },
        "confidence": {
            "value": 0.20,
            "band": "LOW",
            "limiting_reason": "WEAK_EVIDENCE",
            "factors": {},
        },
        "requires_human_review": True,
    }
    out1 = AssessmentOut(**data)
    out2 = AssessmentOut(**data)

    assert isinstance(out1.assessment_id, UUID)
    assert isinstance(out2.assessment_id, UUID)
    assert out1.assessment_id != out2.assessment_id


@pytest.mark.asyncio
async def test_assessment_out_from_domain_service_result() -> None:
    """Verify AssessmentOut can validate and deserialize from pure domain AssessmentResult."""
    service = AssessmentService()
    domain_result = await service.assess(
        title="Payment Service Down",
        description="Transactions are completely failing with HTTP 500 error code.",
        evidence="Datadog alert #100: 5xx rate at 45% for 15 minutes.",
        actions_taken="Attempted restart of worker containers.",
    )

    # Validate directly from domain result using Pydantic model_validate
    dto = AssessmentOut.model_validate(domain_result)

    assert isinstance(dto.assessment_id, UUID)
    assert dto.status == "assessed"
    assert 0.0 <= dto.report_quality.score <= 100.0
    assert len(dto.report_quality.dimensions) == 5
    assert dto.severity.level in {SeverityLevel.SEV1, SeverityLevel.SEV2, SeverityLevel.SEV3, SeverityLevel.SEV4}
    assert 0.0 <= dto.confidence.value <= 1.0
    assert isinstance(dto.requires_human_review, bool)

    # Valid JSON schema output
    json_output = dto.model_dump_json()
    assert "SEV" in json_output


# ==============================================================================
# 4. Settings Configuration Tests
# ==============================================================================


def test_settings_default_instantiation() -> None:
    """Verify Settings singleton and default configuration values."""
    assert settings.app_name == "AI Incident Judgment & Scoring Service"
    assert settings.environment == "development"
    assert settings.llm_provider == "fake"
    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.openai_timeout_seconds == 30.0
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.gemini_timeout_seconds == 30.0
    assert settings.cors_origins == ["*"]


def test_settings_constructor_overrides() -> None:
    """Verify Settings supports direct parameter overrides."""
    custom = Settings(
        app_name="Custom Judgment Service",
        environment="production",
        llm_provider="openai",
        openai_api_key="sk-test-key-12345",
        openai_model="gpt-4o",
        openai_timeout_seconds=60.0,
        gemini_api_key="gemini-key-12345",
        gemini_model="gemini-3.8-flash",
        gemini_timeout_seconds=45.0,
        cors_origins=["https://dashboard.internal.company.com"],
    )
    assert custom.app_name == "Custom Judgment Service"
    assert custom.environment == "production"
    assert custom.llm_provider == "openai"
    assert custom.openai_api_key == "sk-test-key-12345"
    assert custom.openai_model == "gpt-4o"
    assert custom.openai_timeout_seconds == 60.0
    assert custom.gemini_api_key == "gemini-key-12345"
    assert custom.gemini_model == "gemini-3.8-flash"
    assert custom.gemini_timeout_seconds == 45.0
    assert custom.cors_origins == ["https://dashboard.internal.company.com"]


def test_settings_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Settings respects environment variables."""
    monkeypatch.setenv("APP_NAME", "Environment Assessment API")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("LLM_PROVIDER", "OPENAI")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-999")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key-777")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "50.0")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://app.example.com")

    env_settings = Settings()
    assert env_settings.app_name == "Environment Assessment API"
    assert env_settings.environment == "staging"
    assert env_settings.llm_provider == "openai"
    assert env_settings.openai_api_key == "sk-env-key-999"
    assert env_settings.openai_model == "gpt-4o"
    assert env_settings.openai_timeout_seconds == 45.5
    assert env_settings.gemini_api_key == "gemini-env-key-777"
    assert env_settings.gemini_model == "gemini-3.5-flash-lite"
    assert env_settings.gemini_timeout_seconds == 50.0
    assert env_settings.cors_origins == ["http://localhost:3000", "https://app.example.com"]


def test_settings_gemini_provider_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Settings supports GEMINI as LLM_PROVIDER."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    env_settings = Settings()
    assert env_settings.llm_provider == "gemini"
    assert env_settings.gemini_api_key == "gemini-test-key"


def test_settings_cors_origins_json_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Settings parses JSON array format for CORS_ORIGINS."""
    monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
    env_settings = Settings()
    assert env_settings.cors_origins == ["http://localhost:3000", "http://localhost:8080"]


def test_settings_rejects_invalid_llm_provider() -> None:
    """Verify Settings validates and rejects unsupported LLM providers."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(llm_provider="unsupported_provider")
    assert any("llm_provider" in str(err["loc"]) for err in exc_info.value.errors())
