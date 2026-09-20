"""Comprehensive End-to-End API Integration Tests for app.main.

Verifies the complete FastAPI application factory, lifespan management,
CORS middleware, root redirect, OpenAPI documentation routing,
v1 API endpoints (/api/v1/*), input validation (HTTP 422),
and universal exception handling (HTTP 500).
"""

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app.api.dependencies import get_assessment_service
from app.api.v1.schemas import AssessmentOut
from app.judgment.rubric import QUALITY_WEIGHTS, RUBRIC_VERSION
from app.judgment.scoring import (
    FLOOR_DATA_LOSS_OR_BREACH_SEV1,
    FLOOR_TOTAL_OUTAGE_SEV2,
)
from app.judgment.service import AssessmentService
from app.main import app, lifespan


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient with server exceptions unraised for handler assertions."""
    return TestClient(app, raise_server_exceptions=False)


@pytest_asyncio.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    """Provide an asynchronous httpx client backed by ASGITransport."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ==============================================================================
# 1. Root Redirect and Documentation Tests
# ==============================================================================


def test_root_redirects_to_docs(client: TestClient) -> None:
    """Verify GET / returns HTTP 307 redirect pointing to /docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


@pytest.mark.asyncio
async def test_root_redirects_to_docs_async(async_client: httpx.AsyncClient) -> None:
    """Verify GET / returns HTTP 307 redirect using httpx AsyncClient."""
    response = await async_client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_root_redirect_follow(client: TestClient) -> None:
    """Verify following GET / redirect resolves to /docs OpenAPI UI with HTTP 200."""
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert "Swagger UI" in response.text or "swagger-ui" in response.text.lower()


# ==============================================================================
# 2. Health Probe Endpoint Tests
# ==============================================================================


def test_api_v1_health_returns_200(client: TestClient) -> None:
    """Verify GET /api/v1/health returns HTTP 200 with status='healthy' and rubric version."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["rubric_version"] == RUBRIC_VERSION


@pytest.mark.asyncio
async def test_api_v1_health_returns_200_async(async_client: httpx.AsyncClient) -> None:
    """Verify GET /api/v1/health returns HTTP 200 with status='healthy' using httpx AsyncClient."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["rubric_version"] == RUBRIC_VERSION


# ==============================================================================
# 3. Rubric Inspection Endpoint Tests
# ==============================================================================


def test_api_v1_rubric_returns_weights_and_floors(client: TestClient) -> None:
    """Verify GET /api/v1/rubric returns active rubric version, dimension weights, and policy floors."""
    response = client.get("/api/v1/rubric")
    assert response.status_code == 200

    data = response.json()
    assert data["rubric_version"] == RUBRIC_VERSION
    assert "weights" in data
    assert "policy_floors" in data

    # Verify quality dimension weights match domain definitions
    weights = data["weights"]
    assert weights["clarity"] == QUALITY_WEIGHTS["CLARITY"]
    assert weights["evidence_strength"] == QUALITY_WEIGHTS["EVIDENCE_STRENGTH"]
    assert weights["impact_articulation"] == QUALITY_WEIGHTS["IMPACT_ARTICULATION"]
    assert weights["reproducibility"] == QUALITY_WEIGHTS["REPRODUCIBILITY"]
    assert weights["action_context"] == QUALITY_WEIGHTS["ACTION_CONTEXT"]
    assert pytest.approx(sum(weights.values()), rel=1e-5) == 1.0

    # Verify policy floors
    floors = data["policy_floors"]
    assert FLOOR_DATA_LOSS_OR_BREACH_SEV1 in floors
    assert FLOOR_TOTAL_OUTAGE_SEV2 in floors
    assert len(floors) == 2


@pytest.mark.asyncio
async def test_api_v1_rubric_returns_weights_and_floors_async(
    async_client: httpx.AsyncClient,
) -> None:
    """Verify GET /api/v1/rubric using httpx AsyncClient."""
    response = await async_client.get("/api/v1/rubric")
    assert response.status_code == 200
    data = response.json()
    assert data["rubric_version"] == RUBRIC_VERSION
    assert len(data["weights"]) == 5
    assert len(data["policy_floors"]) == 2


# ==============================================================================
# 4. Assessment Endpoint Tests (Realistic Incident Payload)
# ==============================================================================


def test_post_assessments_realistic_payload_with_actions_alias(client: TestClient) -> None:
    """Verify POST /api/v1/assessments with realistic incident payload using 'actions' alias."""
    payload = {
        "title": "Payment gateway timeout 504",
        "description": "Users experiencing checkout failures on payment step.",
        "impact": "Roughly 15% of transactions failing.",
        "evidence": "Datadog alert #401: HTTP 504 gateway timeout spiked to 15%.",
        "actions": "Restarted payment pods.",
    }
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "assessment_id" in data
    assessment_id = UUID(data["assessment_id"])
    assert isinstance(assessment_id, UUID)

    assert data["report_quality"]["score"] > 0
    assert data["confidence"]["value"] > 0
    assert data["status"] == "assessed"

    # Strict schema validation with Pydantic
    assessment = AssessmentOut.model_validate(data)
    assert assessment.report_quality.score > 0
    assert assessment.confidence.value > 0


def test_post_assessments_realistic_payload_with_actions_taken(client: TestClient) -> None:
    """Verify POST /api/v1/assessments with realistic incident payload using 'actions_taken'."""
    payload = {
        "title": "Payment gateway timeout 504",
        "description": "Users experiencing checkout failures on payment step.",
        "impact": "Roughly 15% of transactions failing.",
        "evidence": "Datadog alert #401: HTTP 504 gateway timeout spiked to 15%.",
        "actions_taken": "Restarted payment pods.",
    }
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 200

    data = response.json()
    assessment_id = UUID(data["assessment_id"])
    assert isinstance(assessment_id, UUID)
    assert data["report_quality"]["score"] > 0
    assert data["confidence"]["value"] > 0


@pytest.mark.asyncio
async def test_post_assessments_realistic_payload_async(
    async_client: httpx.AsyncClient,
) -> None:
    """Verify POST /api/v1/assessments using httpx AsyncClient."""
    payload = {
        "title": "Payment gateway timeout 504",
        "description": "Users experiencing checkout failures on payment step.",
        "impact": "Roughly 15% of transactions failing.",
        "evidence": "Datadog alert #401: HTTP 504 gateway timeout spiked to 15%.",
        "actions": "Restarted payment pods.",
    }
    response = await async_client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert UUID(data["assessment_id"])
    assert data["report_quality"]["score"] > 0
    assert data["confidence"]["value"] > 0


# ==============================================================================
# 5. Assessment Endpoint Validation Tests (HTTP 422)
# ==============================================================================


def test_post_assessments_missing_title_returns_422(client: TestClient) -> None:
    """Verify POST /api/v1/assessments rejects payload missing title with HTTP 422."""
    payload = {
        "description": "Users experiencing checkout failures on payment step.",
        "impact": "Roughly 15% of transactions failing.",
        "evidence": "Datadog alert #401: HTTP 504 gateway timeout spiked to 15%.",
        "actions": "Restarted payment pods.",
    }
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 422

    errors = response.json().get("detail", [])
    assert any(err.get("loc") == ["body", "title"] and err.get("type") == "missing" for err in errors)


def test_post_assessments_empty_description_returns_422(client: TestClient) -> None:
    """Verify POST /api/v1/assessments rejects empty or too-short description with HTTP 422."""
    payload = {
        "title": "Payment gateway timeout 504",
        "description": "",
    }
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 422

    errors = response.json().get("detail", [])
    assert any("description" in str(err.get("loc")) for err in errors)


def test_post_assessments_short_description_returns_422(client: TestClient) -> None:
    """Verify POST /api/v1/assessments rejects description under 5 characters with HTTP 422."""
    payload = {
        "title": "Payment gateway timeout 504",
        "description": "Down",
    }
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 422

    errors = response.json().get("detail", [])
    assert any("description" in str(err.get("loc")) for err in errors)


def test_post_assessments_empty_body_returns_422(client: TestClient) -> None:
    """Verify POST /api/v1/assessments rejects empty payload with HTTP 422."""
    response = client.post("/api/v1/assessments", json={})
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert len(errors) >= 2  # Both title and description missing


# ==============================================================================
# 6. Universal Exception Handler Tests (HTTP 500)
# ==============================================================================


def test_universal_exception_handler_structured_json_response(client: TestClient) -> None:
    """Verify unhandled service exception triggers universal handler returning structured 500 JSON."""
    mock_service = MagicMock(spec=AssessmentService)
    error_message = "Unexpected distributed cache failure during pipeline execution"
    mock_service.assess = AsyncMock(side_effect=RuntimeError(error_message))

    app.dependency_overrides[get_assessment_service] = lambda: mock_service

    try:
        payload = {
            "title": "Payment gateway timeout 504",
            "description": "Users experiencing checkout failures on payment step.",
        }
        response = client.post("/api/v1/assessments", json=payload)
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "INTERNAL_SERVER_ERROR"
        assert data["error"]["message"] == error_message
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_universal_exception_handler_async_client(
    async_client: httpx.AsyncClient,
) -> None:
    """Verify universal exception handler with httpx AsyncClient."""
    mock_service = MagicMock(spec=AssessmentService)
    error_message = "Catastrophic threadpool exhaustion"
    mock_service.assess = AsyncMock(side_effect=Exception(error_message))

    app.dependency_overrides[get_assessment_service] = lambda: mock_service

    try:
        payload = {
            "title": "Payment gateway timeout 504",
            "description": "Users experiencing checkout failures on payment step.",
        }
        response = await async_client.post("/api/v1/assessments", json=payload)
        assert response.status_code == 500

        data = response.json()
        assert data == {
            "error": {
                "type": "INTERNAL_SERVER_ERROR",
                "message": error_message,
            }
        }
    finally:
        app.dependency_overrides.clear()


# ==============================================================================
# 7. Middleware & Lifespan Integration Tests
# ==============================================================================


def test_cors_middleware_headers(client: TestClient) -> None:
    """Verify CORS middleware responds to cross-origin requests with configured headers."""
    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] in {"*", "http://localhost:3000"}


def test_cors_preflight_options(client: TestClient) -> None:
    """Verify CORS middleware handles OPTIONS preflight request correctly."""
    response = client.options(
        "/api/v1/assessments",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


@pytest.mark.asyncio
async def test_lifespan_context_manager(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify application lifespan manager executes startup and shutdown actions."""
    async with lifespan(app):
        pass

    captured = capsys.readouterr()
    assert "Starting AI Incident Judgment & Scoring Service" in captured.out
    assert "Rubric: 1.0.0" in captured.out
    assert "Shutting down application..." in captured.out
