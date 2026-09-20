"""API Version 1 Router for Incident Judgment and Scoring.

Exposes REST endpoints for incident report assessment, rubric inspection,
and liveness/readiness health probes.
"""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_assessment_service
from app.api.v1.schemas import AssessmentOut, IncidentInput
from app.judgment.exceptions import IncidentJudgmentError, LLMProviderError
from app.judgment.rubric import QUALITY_WEIGHTS, RUBRIC_VERSION
from app.judgment.scoring import (
    FLOOR_DATA_LOSS_OR_BREACH_SEV1,
    FLOOR_TOTAL_OUTAGE_SEV2,
)
from app.judgment.service import AssessmentService

router = APIRouter(prefix="/v1", tags=["Incidents"])


@router.post(
    "/assessments",
    response_model=AssessmentOut,
    status_code=status.HTTP_200_OK,
    summary="Assess an incident report",
)
async def assess_incident(
    payload: IncidentInput,
    service: AssessmentService = Depends(get_assessment_service),
) -> AssessmentOut:
    """Assess an incident report through the automated AI judgment pipeline.

    Receives a validated incident payload, invokes the domain AssessmentService,
    and returns a serialized AssessmentOut schema response.
    """
    try:
        domain_result = await service.assess(
            title=payload.title,
            description=payload.description,
            impact=payload.impact,
            evidence=payload.evidence,
            actions_taken=payload.actions_taken,
        )
        return AssessmentOut.model_validate(domain_result)
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM Provider Unavailable: {str(exc)}",
        ) from exc
    except IncidentJudgmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Incident Judgment Error: {str(exc)}",
        ) from exc


@router.get(
    "/rubric",
    status_code=status.HTTP_200_OK,
    summary="Inspect active rubric, weights, and policy floors",
)
async def get_rubric() -> dict[str, Any]:
    """Inspect active rubric version, dimension weights, and deterministic policy floors."""
    weights = {
        (dim.value.lower() if hasattr(dim, "value") else str(dim).lower()): weight
        for dim, weight in QUALITY_WEIGHTS.items()
    }
    return {
        "rubric_version": RUBRIC_VERSION,
        "weights": weights,
        "policy_floors": [
            FLOOR_DATA_LOSS_OR_BREACH_SEV1,
            FLOOR_TOTAL_OUTAGE_SEV2,
        ],
    }


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Liveness and readiness probe",
)
async def health() -> dict[str, str]:
    """Liveness and readiness probe reporting API status and active rubric version."""
    return {
        "status": "healthy",
        "rubric_version": RUBRIC_VERSION,
    }
