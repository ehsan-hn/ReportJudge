"""Mathematical Confidence Calibrator.

Enforces deterministic Python arithmetic to address the LLM overconfidence problem.
If evidence is weak or critical information is missing, confidence is deterministically
capped and penalized, regardless of how confident the LLM claims to be.
Zero framework or HTTP dependencies.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from .contracts import LLMAssessment
from .rubric import QualityDimension

LIMITING_REASON_WEAK_EVIDENCE: str = "WEAK_OR_UNVERIFIED_EVIDENCE"
LIMITING_REASON_GAPS: str = "NUMEROUS_INFORMATION_GAPS"

BAND_LOW: str = "LOW"
BAND_MEDIUM: str = "MEDIUM"
BAND_HIGH: str = "HIGH"

MIN_CONFIDENCE: float = 0.05
MAX_CONFIDENCE: float = 0.95


def compute_confidence(
    assessment: LLMAssessment,
    provided_fields_count: int,
    total_optional_fields: int = 3,
) -> dict[str, Any]:
    """Compute deterministic mathematical confidence calibration for an incident assessment.

    Directly mitigates the LLM overconfidence problem by capping and penalizing confidence
    based on objective evidence strength, missing information gaps, and optional field coverage,
    independent of self-reported LLM confidence.

    Args:
        assessment: The validated LLMAssessment output contract.
        provided_fields_count: Number of optional incident fields supplied in the report.
        total_optional_fields: Total optional incident fields considered (default 3).

    Returns:
        dict[str, Any]: Calibrated confidence value, classification band, limiting reason,
        and factor breakdown.
    """
    # 1. Base Model Confidence:
    # Mean of all 5 dimension confidences plus severity_signals.confidence
    dim_confidences = [
        assessment.dimensions[dim].confidence for dim in QualityDimension
    ]
    all_confidences = dim_confidences + [assessment.severity_signals.confidence]
    model_conf = mean(all_confidences)

    # 2. Hard Evidence Ceiling:
    # Maps evidence_score (0..5) to a max ceiling of 0.30..1.00
    evidence_score = assessment.dimensions[QualityDimension.EVIDENCE_STRENGTH].score
    evidence_ceiling = 0.30 + (0.14 * evidence_score)

    # 3. Field Coverage Scalar:
    # Scales from 0.75 to 1.00 based on optional field presence
    safe_total_fields = max(1, total_optional_fields)
    coverage_ratio = max(0.0, min(1.0, provided_fields_count / safe_total_fields))
    coverage_factor = 0.75 + (0.25 * coverage_ratio)

    # 4. Missing Information Gap Penalty:
    # Deducts up to 0.25 (0.04 per missing information item)
    gap_penalty = min(0.25, 0.04 * len(assessment.missing_information))

    # 5. Bounded Confidence Calculation:
    # Hard clamp between 0.05 and 0.95
    raw_conf = (model_conf * evidence_ceiling * coverage_factor) - gap_penalty
    final_value = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, raw_conf))
    rounded_value = round(final_value, 2)

    # 6. Confidence Band Classification:
    if rounded_value < 0.45:
        band = BAND_LOW
    elif rounded_value < 0.70:
        band = BAND_MEDIUM
    else:
        band = BAND_HIGH

    # 7. Limiting Reason Attribution:
    if evidence_ceiling < 0.60:
        limiting_reason = LIMITING_REASON_WEAK_EVIDENCE
    elif gap_penalty >= 0.15:
        limiting_reason = LIMITING_REASON_GAPS
    else:
        limiting_reason = None

    # 8. Return structure
    return {
        "value": rounded_value,
        "band": band,
        "limiting_reason": limiting_reason,
        "factors": {
            "model_self_reported": round(model_conf, 2),
            "evidence_ceiling": round(evidence_ceiling, 2),
            "coverage_factor": round(coverage_factor, 2),
            "gap_penalty": round(gap_penalty, 2),
        },
    }
