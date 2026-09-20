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
from app.judgment.confidence import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    LIMITING_REASON_GAPS,
    LIMITING_REASON_WEAK_EVIDENCE,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    compute_confidence,
)
from app.judgment.scoring import (
    FLOOR_DATA_LOSS_OR_BREACH_SEV1,
    FLOOR_TOTAL_OUTAGE_SEV2,
    compute_quality_score,
    compute_severity,
)
from app.judgment.guardrails import (
    UNTRUSTED_INSTRUCTION_NOTICE,
    sanitize_text,
    wrap_untrusted_input,
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
    "LIMITING_REASON_WEAK_EVIDENCE",
    "LIMITING_REASON_GAPS",
    "BAND_LOW",
    "BAND_MEDIUM",
    "BAND_HIGH",
    "MIN_CONFIDENCE",
    "MAX_CONFIDENCE",
    "compute_confidence",
    "UNTRUSTED_INSTRUCTION_NOTICE",
    "sanitize_text",
    "wrap_untrusted_input",
]
