"""Domain package for AI Incident Judgment and Scoring.

This package contains pure business logic, rubric definitions, domain models,
and domain exceptions with ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from app.judgment.contracts import (
    DimensionSignal,
    LLMAssessment,
    SeveritySignals,
)
from app.judgment.exceptions import (
    IncidentJudgmentError,
    LLMProviderError,
    SchemaValidationError,
    UntrustedInputError,
)
from app.judgment.rubric import (
    PROMPT_VERSION,
    QUALITY_WEIGHTS,
    RUBRIC_VERSION,
    BusinessCriticality,
    ImpactScope,
    MissingInfoCode,
    QualityDimension,
    SeverityLevel,
    TimeSensitivity,
)
from app.judgment.scoring import (
    FLOOR_DATA_LOSS_OR_BREACH_SEV1,
    FLOOR_TOTAL_OUTAGE_SEV2,
    compute_quality_score,
    compute_severity,
)

__all__ = [
    "RUBRIC_VERSION",
    "PROMPT_VERSION",
    "QualityDimension",
    "QUALITY_WEIGHTS",
    "ImpactScope",
    "BusinessCriticality",
    "TimeSensitivity",
    "SeverityLevel",
    "MissingInfoCode",
    "IncidentJudgmentError",
    "LLMProviderError",
    "SchemaValidationError",
    "UntrustedInputError",
    "DimensionSignal",
    "SeveritySignals",
    "LLMAssessment",
    "FLOOR_DATA_LOSS_OR_BREACH_SEV1",
    "FLOOR_TOTAL_OUTAGE_SEV2",
    "compute_quality_score",
    "compute_severity",
]
