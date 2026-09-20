"""API Data Transfer Objects (DTOs) & Validation Schemas.

Enforces strict input validation boundaries at the API gateway layer
and serializes domain evaluation results into OpenAPI-compliant JSON schemas.
"""

from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.judgment.rubric import MissingInfoCode, SeverityLevel


# ==============================================================================
# Request Schemas
# ==============================================================================


class IncidentInput(BaseModel):
    """Incident report input payload submitted for automated AI judgment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Brief summary or title of the incident.",
        json_schema_extra={"example": "Checkout API returning 500"},
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=8000,
        description="Detailed narrative explaining what occurred and observed symptoms.",
        json_schema_extra={"example": "Users cannot complete purchases on payment step."},
    )
    impact: str | None = Field(
        default=None,
        max_length=2000,
        description="User, business, or operational impact.",
        json_schema_extra={"example": "Roughly 20% of European traffic failing."},
    )
    evidence: str | None = Field(
        default=None,
        max_length=4000,
        description="Telemetry, logs, metric charts, or alert links confirming the issue.",
        json_schema_extra={"example": "Datadog alert #502, 5xx rate spiked to 18%."},
    )
    actions_taken: str | None = Field(
        default=None,
        max_length=2000,
        description="Troubleshooting, rollbacks, or mitigation steps performed.",
        json_schema_extra={"example": "Scaled pods to 12; no resolution."},
    )

    @model_validator(mode="before")
    @classmethod
    def handle_actions_alias(cls, data: Any) -> Any:
        """Support 'actions' as a valid alias for 'actions_taken'."""
        if isinstance(data, dict) and "actions" in data and "actions_taken" not in data:
            data = dict(data)
            data["actions_taken"] = data.pop("actions")
        return data


# ==============================================================================
# Response Sub-Schemas
# ==============================================================================


class DimensionDetail(BaseModel):
    """Evaluation breakdown and scoring for a single quality rubric dimension."""

    model_config = ConfigDict(from_attributes=True)

    dimension: str = Field(
        ...,
        description="Name of the quality dimension evaluated.",
        json_schema_extra={"example": "CLARITY"},
    )
    score: int = Field(
        ...,
        ge=0,
        le=5,
        description="Ordinal score from 0 (absent/misleading) to 5 (exemplary).",
        json_schema_extra={"example": 4},
    )
    weight: float = Field(
        ...,
        description="Dimension weight in overall quality score calculation.",
        json_schema_extra={"example": 0.20},
    )
    justification: str = Field(
        ...,
        description="Model justification conditioned prior to score assignment.",
        json_schema_extra={"example": "Incident timeline and symptoms clearly articulated."},
    )
    confidence: float = Field(
        ...,
        description="Confidence signal for this specific dimension.",
        json_schema_extra={"example": 0.9},
    )


class QualityBreakdown(BaseModel):
    """Aggregated report quality evaluation with dimension details."""

    model_config = ConfigDict(from_attributes=True)

    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall weighted quality score between 0.0 and 100.0.",
        json_schema_extra={"example": 82.5},
    )
    band: str = Field(
        ...,
        description="Qualitative performance band ('EXCELLENT', 'GOOD', 'FAIR', 'POOR').",
        json_schema_extra={"example": "GOOD"},
    )
    dimensions: list[DimensionDetail] = Field(
        ...,
        description="Breakdown of individual quality dimensions.",
    )


class SeverityBreakdown(BaseModel):
    """Deterministic severity classification and rule execution trace."""

    model_config = ConfigDict(from_attributes=True)

    level: SeverityLevel = Field(
        ...,
        description="Final severity classification (SEV1, SEV2, SEV3, SEV4).",
        json_schema_extra={"example": "SEV2"},
    )
    index: float = Field(
        ...,
        description="Continuous severity index before policy floor adjustments.",
        json_schema_extra={"example": 2.15},
    )
    applied_rules: list[str] = Field(
        default_factory=list,
        description="List of deterministic policy floor rules triggered during assessment.",
        json_schema_extra={"example": []},
    )


class ConfidenceBreakdown(BaseModel):
    """Calibrated confidence evaluation and limiting factors."""

    model_config = ConfigDict(from_attributes=True)

    value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Bounded, calibrated confidence score between 0.0 and 1.0.",
        json_schema_extra={"example": 0.85},
    )
    band: str = Field(
        ...,
        description="Confidence tier ('HIGH', 'MEDIUM', 'LOW').",
        json_schema_extra={"example": "HIGH"},
    )
    limiting_reason: str | None = Field(
        default=None,
        description="Explanation if confidence was bounded or capped by a ceiling rule.",
        json_schema_extra={"example": None},
    )
    factors: dict[str, float] = Field(
        default_factory=dict,
        description="Decomposed mathematical factors contributing to confidence score.",
        json_schema_extra={"example": {"model_self_reported": 0.9, "coverage_factor": 1.0}},
    )


# ==============================================================================
# Main Response Schema
# ==============================================================================


class AssessmentOut(BaseModel):
    """Complete incident evaluation and scoring response DTO."""

    model_config = ConfigDict(from_attributes=True)

    assessment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this assessment execution.",
        json_schema_extra={"example": "c9bf9e57-1685-4c89-bafb-ff5af830be8a"},
    )
    status: str = Field(
        ...,
        description="Assessment status outcome (e.g. 'assessed', 'not_an_incident_report', 'insufficient_input').",
        json_schema_extra={"example": "assessed"},
    )
    report_quality: QualityBreakdown = Field(
        ...,
        description="Deterministic quality score and dimension breakdown.",
    )
    severity: SeverityBreakdown = Field(
        ...,
        description="Deterministic severity level and policy rule trail.",
    )
    confidence: ConfidenceBreakdown = Field(
        ...,
        description="Calibrated confidence and bounding factor breakdown.",
    )
    missing_information: list[MissingInfoCode] = Field(
        default_factory=list,
        description="Standardized missing incident information tags.",
        json_schema_extra={"example": ["NO_REPRO_STEPS"]},
    )
    missing_information_notes: str = Field(
        default="",
        description="Elaborated notes on missing incident context.",
        json_schema_extra={"example": "No reproduction steps provided for payment checkout failure."},
    )
    summary_explanation: str = Field(
        default="",
        description="Summary explanation synthesizing the incident evaluation.",
        json_schema_extra={"example": "Critical payment outage with high user impact; reproducible mitigation pending."},
    )
    requires_human_review: bool = Field(
        ...,
        description="Indicates whether this evaluation requires escalation or human verification.",
        json_schema_extra={"example": False},
    )
