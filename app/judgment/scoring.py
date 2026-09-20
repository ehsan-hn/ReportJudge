"""Deterministic Scoring Engine & Policy Floors.

This module enforces pure Python arithmetic, normalization, band classification,
and hard policy floors with zero LLM arithmetic and zero HTTP/framework dependencies.
"""

from .contracts import DimensionSignal, SeveritySignals
from .rubric import (
    QUALITY_WEIGHTS,
    BusinessCriticality,
    ImpactScope,
    QualityDimension,
    SeverityLevel,
)

FLOOR_DATA_LOSS_OR_BREACH_SEV1: str = "FLOOR_DATA_LOSS_OR_BREACH_SEV1"
FLOOR_TOTAL_OUTAGE_SEV2: str = "FLOOR_TOTAL_OUTAGE_SEV2"


def compute_quality_score(
    dims: dict[QualityDimension, DimensionSignal],
) -> tuple[float, str]:
    """Calculate deterministic weighted quality score and corresponding quality band.

    Args:
        dims: Mapping of all QualityDimension enums to their DimensionSignal outputs.

    Returns:
        tuple[float, str]: Normalized score (0.0 to 100.0) rounded to 1 decimal place,
        and the associated quality band ("POOR", "WEAK", "ADEQUATE", or "STRONG").

    Raises:
        ValueError: If any required quality dimension is missing from dims.
    """
    missing = set(QualityDimension) - set(dims.keys())
    if missing:
        raise ValueError(
            f"Missing quality dimensions: {sorted(d.value if isinstance(d, QualityDimension) else str(d) for d in missing)}"
        )

    raw_weighted_sum = sum(QUALITY_WEIGHTS[d] * dims[d].score for d in QualityDimension)
    score = round((raw_weighted_sum / 5.0) * 100.0, 1)

    if score < 35.0:
        band = "POOR"
    elif score < 55.0:
        band = "WEAK"
    elif score < 75.0:
        band = "ADEQUATE"
    else:
        band = "STRONG"

    return score, band


def compute_severity(
    signals: SeveritySignals,
) -> tuple[SeverityLevel, float, list[str]]:
    """Compute incident severity level with deterministic policy floors.

    Calculates a continuous baseline severity index (0.0 to 4.0), maps to an
    initial baseline severity level, and applies hard non-negotiable policy floors.

    Args:
        signals: Ordinal signals representing impact scope, business criticality,
                 and time sensitivity.

    Returns:
        tuple[SeverityLevel, float, list[str]]:
            - Final SeverityLevel (SEV1 through SEV4)
            - Continuous severity index rounded to 2 decimal places
            - List of string identifiers for any applied policy floors
    """
    index = (
        0.45 * signals.impact_scope
        + 0.35 * signals.business_criticality
        + 0.20 * (signals.time_sensitivity / 3.0 * 4.0)
    )

    if index < 1.0:
        level = SeverityLevel.SEV4
    elif index < 2.0:
        level = SeverityLevel.SEV3
    elif index < 3.0:
        level = SeverityLevel.SEV2
    else:
        level = SeverityLevel.SEV1

    applied_floors: list[str] = []

    # Policy Floor 1 (FLOOR_DATA_LOSS_OR_BREACH_SEV1):
    # If business criticality indicates data loss or breach, elevate/set to SEV1.
    if signals.business_criticality >= BusinessCriticality.DATA_LOSS_OR_BREACH:
        level = SeverityLevel.SEV1
        applied_floors.append(FLOOR_DATA_LOSS_OR_BREACH_SEV1)

    # Policy Floor 2 (FLOOR_TOTAL_OUTAGE_SEV2):
    # If impact scope is ALL_USERS, business criticality is at least CORE_FLOW_IMPAIRED,
    # and baseline level is SEV3 or SEV4, elevate to SEV2. Never downgrades a natural SEV1.
    if (
        signals.impact_scope >= ImpactScope.ALL_USERS
        and signals.business_criticality >= BusinessCriticality.CORE_FLOW_IMPAIRED
        and level in (SeverityLevel.SEV3, SeverityLevel.SEV4)
    ):
        level = SeverityLevel.SEV2
        applied_floors.append(FLOOR_TOTAL_OUTAGE_SEV2)

    return level, round(index, 2), applied_floors
