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
]
