"""Integration Tests for FastAPI Router Endpoints & Dependency Injection.

Verifies HTTP routing, dependency injection decoupling, schema validation,
status codes, and error mapping for:
- POST /v1/assessments
- GET /v1/rubric
- GET /v1/health
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_assessment_service
from app.api.v1.router import router
from app.api.v1.schemas import AssessmentOut
from app.judgment.exceptions import IncidentJudgmentError, LLMProviderError
from app.judgment.rubric import QUALITY_WEIGHTS, RUBRIC_VERSION
from app.judgment.scoring import (
    FLOOR_DATA_LOSS_OR_BREACH_SEV1,
    FLOOR_TOTAL_OUTAGE_SEV2,
)
from app.judgment.service import AssessmentService


@pytest.fixture
def app() -> FastAPI:
    """Create a lightweight test FastAPI application mounting the v1 router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Provide a TestClient bound to the test FastAPI application."""
    return TestClient(app)


# ==============================================================================
# 1. Health Probe Endpoint Tests
# ==============================================================================


def test_health_endpoint_returns_200(client: TestClient) -> None:
    """Verify GET /v1/health returns HTTP 200 with healthy status and rubric version."""
    response = client.get("/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["rubric_version"] == RUBRIC_VERSION
    assert data == {
        "status": "healthy",
        "rubric_version": "1.0.0",
    }


# ==============================================================================
# 2. Rubric Inspection Endpoint Tests
# ==============================================================================


def test_rubric_endpoint_returns_weights_and_floors(client: TestClient) -> None:
    """Verify GET /v1/rubric returns active rubric version, dimension weights, and policy floors."""
    response = client.get("/v1/rubric")
    assert response.status_code == 200

    data = response.json()
    assert data["rubric_version"] == RUBRIC_VERSION
    assert "weights" in data
    assert "policy_floors" in data

    # Verify quality dimension weights
    weights = data["weights"]
    assert weights["clarity"] == QUALITY_WEIGHTS["CLARITY"]
    assert weights["evidence_strength"] == QUALITY_WEIGHTS["EVIDENCE_STRENGTH"]
    assert weights["impact_articulation"] == QUALITY_WEIGHTS["IMPACT_ARTICULATION"]
    assert weights["reproducibility"] == QUALITY_WEIGHTS["REPRODUCIBILITY"]
    assert weights["action_context"] == QUALITY_WEIGHTS["ACTION_CONTEXT"]
    assert pytest.approx(sum(weights.values()), rel=1e-5) == 1.0

    # Verify deterministic policy floors
    floors = data["policy_floors"]
    assert FLOOR_DATA_LOSS_OR_BREACH_SEV1 in floors
    assert FLOOR_TOTAL_OUTAGE_SEV2 in floors
    assert len(floors) == 2


# ==============================================================================
# 3. Assessment Endpoint Tests (Valid Payloads)
# ==============================================================================


def test_assessments_endpoint_valid_minimal_payload(client: TestClient) -> None:
    """Verify POST /v1/assessments with minimal valid payload returns 200 and matches AssessmentOut."""
    payload = {
        "title": "Database connection pool exhausted",
        "description": "Backend services throwing 500 error due to connection timeout on main cluster.",
    }
    response = client.post("/v1/assessments", json=payload)
    assert response.status_code == 200

    data = response.json()
    assessment = AssessmentOut.model_validate(data)

    assert isinstance(assessment.assessment_id, UUID)
    assert assessment.status == "assessed"
    assert 0.0 <= assessment.report_quality.score <= 100.0
    assert assessment.report_quality.band in {"STRONG", "ADEQUATE", "WEAK", "POOR"}
    assert len(assessment.report_quality.dimensions) == 5
    assert assessment.severity.level.value in {"SEV1", "SEV2", "SEV3", "SEV4"}
    assert 0.0 <= assessment.confidence.value <= 1.0
    assert assessment.confidence.band in {"HIGH", "MEDIUM", "LOW"}
    assert isinstance(assessment.requires_human_review, bool)


def test_assessments_endpoint_valid_full_payload(client: TestClient) -> None:
    """Verify POST /v1/assessments with complete payload returns 200 and valid schema."""
    payload = {
        "title": "Payment gateway latency spike",
        "description": "Checkout API p99 latency exceeded 5 seconds across EU-West region.",
        "impact": "Approximately 15% of checkout transactions timed out during morning peak.",
        "evidence": "Datadog monitor alert #4412, APM trace shows RDS lock contention.",
        "actions_taken": "Restarted connection pooling proxy and cleared stale client connections.",
    }
    response = client.post("/v1/assessments", json=payload)
    assert response.status_code == 200

    data = response.json()
    assessment = AssessmentOut.model_validate(data)

    assert isinstance(assessment.assessment_id, UUID)
    assert assessment.status == "assessed"
    assert assessment.report_quality.score > 0.0
    assert isinstance(assessment.severity.applied_rules, list)


def test_assessments_endpoint_non_incident_report(client: TestClient) -> None:
    """Verify POST /v1/assessments with non-incident text returns 200 with appropriate status."""
    payload = {
        "title": "Good morning team",
        "description": "Just wanted to say hello and wish everyone a great productive week!",
    }
    response = client.post("/v1/assessments", json=payload)
    assert response.status_code == 200

    data = response.json()
    assessment = AssessmentOut.model_validate(data)

    assert assessment.status == "not_an_incident_report"
    assert assessment.requires_human_review is True
    assert assessment.report_quality.score == 0.0
    assert assessment.report_quality.band == "POOR"


# ==============================================================================
# 4. Assessment Endpoint Tests (Validation & Error Handling)
# ==============================================================================


def test_assessments_endpoint_invalid_title_too_short(client: TestClient) -> None:
    """Verify POST /v1/assessments rejects title under 3 characters with 422."""
    payload = {
        "title": "ab",
        "description": "Valid incident description.",
    }
    response = client.post("/v1/assessments", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("title" in str(err["loc"]) for err in errors)


def test_assessments_endpoint_invalid_description_too_short(client: TestClient) -> None:
    """Verify POST /v1/assessments rejects description under 5 characters with 422."""
    payload = {
        "title": "Valid Incident Title",
        "description": "Bad",
    }
    response = client.post("/v1/assessments", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("description" in str(err["loc"]) for err in errors)


def test_assessments_endpoint_missing_required_fields(client: TestClient) -> None:
    """Verify POST /v1/assessments rejects payloads missing title or description with 422."""
    # Missing description
    response = client.post("/v1/assessments", json={"title": "Valid Title"})
    assert response.status_code == 422

    # Missing title
    response = client.post("/v1/assessments", json={"description": "Valid description long enough"})
    assert response.status_code == 422

    # Completely empty body
    response = client.post("/v1/assessments", json={})
    assert response.status_code == 422


def test_assessments_endpoint_extra_field_forbidden(client: TestClient) -> None:
    """Verify extra='forbid' rejects unrecognized fields with 422."""
    payload = {
        "title": "Valid Incident Title",
        "description": "Valid incident description long enough.",
        "unexpected_field": "disallowed value",
    }
    response = client.post("/v1/assessments", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(err.get("type") == "extra_forbidden" for err in errors)


def test_assessments_endpoint_llm_provider_error_raises_503(app: FastAPI, client: TestClient) -> None:
    """Verify domain LLMProviderError is translated into HTTP 503 Service Unavailable."""
    mock_service = MagicMock(spec=AssessmentService)
    mock_service.assess = AsyncMock(side_effect=LLMProviderError("Connection timeout to OpenAI API"))

    app.dependency_overrides[get_assessment_service] = lambda: mock_service

    try:
        payload = {
            "title": "Kafka Broker Failure",
            "description": "Broker 3 went offline causing partition offline alerts.",
        }
        response = client.post("/v1/assessments", json=payload)
        assert response.status_code == 503
        assert "LLM Provider Unavailable" in response.json()["detail"]
        assert "Connection timeout to OpenAI API" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_assessments_endpoint_general_judgment_error_raises_500(app: FastAPI, client: TestClient) -> None:
    """Verify general domain IncidentJudgmentError is translated into HTTP 500."""
    mock_service = MagicMock(spec=AssessmentService)
    mock_service.assess = AsyncMock(side_effect=IncidentJudgmentError("Unexpected domain rule corruption"))

    app.dependency_overrides[get_assessment_service] = lambda: mock_service

    try:
        payload = {
            "title": "Kafka Broker Failure",
            "description": "Broker 3 went offline causing partition offline alerts.",
        }
        response = client.post("/v1/assessments", json=payload)
        assert response.status_code == 500
        assert "Incident Judgment Error" in response.json()["detail"]
        assert "Unexpected domain rule corruption" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


# ==============================================================================
# 5. Dependency Injection Unit Tests
# ==============================================================================


def test_get_assessment_service_provider() -> None:
    """Verify get_assessment_service returns an AssessmentService and caches singleton."""
    service1 = get_assessment_service()
    service2 = get_assessment_service()

    assert isinstance(service1, AssessmentService)
    assert service1 is service2


def test_get_assessment_service_gemini_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_assessment_service instantiates GeminiLLMClient when llm_provider is gemini."""
    from app.config import settings
    from app.judgment.llm.gemini_client import GeminiLLMClient

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "fake-gemini-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.8-flash")
    monkeypatch.setattr(settings, "gemini_timeout_seconds", 25.0)

    get_assessment_service.cache_clear()
    try:
        service = get_assessment_service()
        assert isinstance(service, AssessmentService)
        assert isinstance(service.llm_client, GeminiLLMClient)
        assert service.llm_client.model == "gemini-3.8-flash"
    finally:
        get_assessment_service.cache_clear()
