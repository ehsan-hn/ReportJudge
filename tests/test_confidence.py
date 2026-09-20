"""Mathematical Confidence Calibrator Boundary & Invariant Tests.

Tests the deterministic bounding, evidence ceilings, optional field coverage,
information gap penalties, band classifications, and limiting reason attributions.
Zero framework or HTTP dependencies.
"""

import pytest

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
from app.judgment.contracts import (
    DimensionSignal,
    LLMAssessment,
    SeveritySignals,
)
from app.judgment.rubric import (
    BusinessCriticality,
    ImpactScope,
    MissingInfoCode,
    QualityDimension,
    TimeSensitivity,
)


def make_assessment(
    model_confidence: float = 0.9,
    evidence_score: int = 4,
    dimension_scores: dict[QualityDimension, int] | None = None,
    dimension_confidences: dict[QualityDimension, float] | None = None,
    severity_confidence: float | None = None,
    missing_information: list[MissingInfoCode] | None = None,
    summary: str = "Test incident assessment.",
) -> LLMAssessment:
    """Helper to construct a valid LLMAssessment with customizable confidence and evidence."""
    dim_scores = dimension_scores or {}
    dim_confs = dimension_confidences or {}

    dims: dict[QualityDimension, DimensionSignal] = {}
    for dim in QualityDimension:
        score = evidence_score if dim == QualityDimension.EVIDENCE_STRENGTH else dim_scores.get(dim, 4)
        conf = dim_confs.get(dim, model_confidence)
        dims[dim] = DimensionSignal(
            justification=f"Assessment justification for {dim.value}",
            evidence_quote=f"Quote for {dim.value}",
            score=score,
            confidence=conf,
        )

    sev_conf = severity_confidence if severity_confidence is not None else model_confidence
    severity = SeveritySignals(
        reasoning="Severity assessment reasoning.",
        impact_scope=ImpactScope.SMALL_SUBSET,
        business_criticality=BusinessCriticality.DEGRADED_UX,
        time_sensitivity=TimeSensitivity.SLOW_DEGRADATION,
        confidence=sev_conf,
    )

    return LLMAssessment(
        is_valid_incident_report=True,
        dimensions=dims,
        severity_signals=severity,
        missing_information=missing_information or [],
        missing_information_notes="Notes on missing info.",
        summary=summary,
    )


# ==============================================================================
# 1. Clamping Assertions
# ==============================================================================


def test_clamping_lower_boundary_zero_model_confidence():
    """Even if base model confidence is 0.0, output cannot drop below 0.05."""
    assessment = make_assessment(
        model_confidence=0.0,
        evidence_score=5,
        missing_information=[],
    )
    result = compute_confidence(assessment, provided_fields_count=3, total_optional_fields=3)
    assert result["value"] == MIN_CONFIDENCE
    assert result["value"] == 0.05
    assert result["band"] == BAND_LOW


def test_clamping_lower_boundary_negative_raw_confidence():
    """Even with zero model confidence and maximum gap penalty (-0.25), output clamps to 0.05."""
    all_gaps = list(MissingInfoCode)  # 8 items -> 0.25 penalty
    assessment = make_assessment(
        model_confidence=0.0,
        evidence_score=0,
        missing_information=all_gaps,
    )
    result = compute_confidence(assessment, provided_fields_count=0, total_optional_fields=3)
    assert result["value"] == 0.05
    assert result["band"] == BAND_LOW


def test_clamping_upper_boundary_perfect_signals():
    """Even if all model confidences are 1.0, evidence is 5, and all fields present, output cannot exceed 0.95."""
    assessment = make_assessment(
        model_confidence=1.0,
        evidence_score=5,
        missing_information=[],
    )
    result = compute_confidence(assessment, provided_fields_count=3, total_optional_fields=3)
    # raw: 1.0 * 1.0 * 1.0 - 0.0 = 1.0 -> clamped to 0.95
    assert result["value"] == MAX_CONFIDENCE
    assert result["value"] == 0.95
    assert result["band"] == BAND_HIGH


# ==============================================================================
# 2. Hard Evidence Ceiling Assertions
# ==============================================================================


def test_evidence_ceiling_at_score_zero():
    """When evidence_strength score is 0, confidence must never exceed 0.30 even with 1.0 self-reported confidence."""
    assessment = make_assessment(
        model_confidence=1.0,
        evidence_score=0,
        missing_information=[],
    )
    # With full coverage (3/3) and 0 gap penalty: raw = 1.0 * 0.30 * 1.0 - 0.0 = 0.30
    result = compute_confidence(assessment, provided_fields_count=3, total_optional_fields=3)
    assert result["value"] <= 0.30
    assert result["value"] == 0.30
    assert result["factors"]["evidence_ceiling"] == 0.30
    assert result["band"] == BAND_LOW


@pytest.mark.parametrize(
    "evidence_score, expected_ceiling",
    [
        (0, 0.30),
        (1, 0.44),
        (2, 0.58),
        (3, 0.72),
        (4, 0.86),
        (5, 1.00),
    ],
)
def test_evidence_ceiling_formula_mapping(evidence_score: int, expected_ceiling: float):
    """Verify evidence_ceiling formula: 0.30 + (0.14 * evidence_score) across all possible scores (0..5)."""
    assessment = make_assessment(evidence_score=evidence_score)
    result = compute_confidence(assessment, provided_fields_count=3, total_optional_fields=3)
    assert result["factors"]["evidence_ceiling"] == expected_ceiling


# ==============================================================================
# 3. Gap Penalty Assertions
# ==============================================================================


def test_gap_penalty_monotonic_depression():
    """Monotonically test that adding MissingInfoCode items continuously depresses confidence (capping at 0.25)."""
    codes = list(MissingInfoCode)  # 8 total distinct codes
    previous_value = 1.0
    previous_penalty = -0.01

    expected_penalties = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.25, 0.25]

    for count in range(len(codes) + 1):
        active_codes = codes[:count]
        assessment = make_assessment(
            model_confidence=0.95,
            evidence_score=5,
            missing_information=active_codes,
        )
        result = compute_confidence(assessment, provided_fields_count=3, total_optional_fields=3)
        current_penalty = result["factors"]["gap_penalty"]
        current_value = result["value"]

        # Penalty matches expected exact value
        assert current_penalty == expected_penalties[count]
        assert current_penalty >= previous_penalty

        # Confidence value strictly decreases until cap (count 7 and 8 have identical 0.25 penalty)
        if count <= 7:
            assert current_value <= previous_value
        if count < 7:
            assert current_value < previous_value
        elif count == 8:
            assert current_value == previous_value

        previous_penalty = current_penalty
        previous_value = current_value


def test_gap_penalty_cap_at_twenty_five():
    """Gap penalty cap strictly stops at 0.25 even with more than 7 missing info items."""
    many_gaps = [MissingInfoCode.NO_TIMELINE] * 10
    assessment = make_assessment(missing_information=many_gaps)
    result = compute_confidence(assessment, provided_fields_count=3, total_optional_fields=3)
    assert result["factors"]["gap_penalty"] == 0.25


# ==============================================================================
# 4. Field Coverage Scalar Assertions
# ==============================================================================


def test_coverage_factor_zero_vs_full():
    """Providing 0 optional fields results in strictly lower confidence than providing all 3 optional fields."""
    assessment = make_assessment(
        model_confidence=0.90,
        evidence_score=4,
        missing_information=[],
    )
    result_zero = compute_confidence(assessment, provided_fields_count=0, total_optional_fields=3)
    result_full = compute_confidence(assessment, provided_fields_count=3, total_optional_fields=3)

    assert result_zero["factors"]["coverage_factor"] == 0.75
    assert result_full["factors"]["coverage_factor"] == 1.00
    assert result_zero["value"] < result_full["value"]


@pytest.mark.parametrize(
    "provided_fields, expected_factor",
    [
        (0, 0.75),
        (1, 0.83),  # 0.75 + 0.25 * (1/3) = 0.8333... -> 0.83
        (2, 0.92),  # 0.75 + 0.25 * (2/3) = 0.9166... -> 0.92
        (3, 1.00),  # 0.75 + 0.25 * (3/3) = 1.00
    ],
)
def test_coverage_factor_scaling(provided_fields: int, expected_factor: float):
    """Test coverage factor calculation across partial and complete field coverage."""
    assessment = make_assessment()
    result = compute_confidence(assessment, provided_fields_count=provided_fields, total_optional_fields=3)
    assert result["factors"]["coverage_factor"] == expected_factor


def test_coverage_factor_default_optional_fields():
    """total_optional_fields defaults to 3 if omitted."""
    assessment = make_assessment()
    result = compute_confidence(assessment, provided_fields_count=3)
    assert result["factors"]["coverage_factor"] == 1.00


def test_coverage_factor_zero_total_optional_fields_safe():
    """total_optional_fields=0 does not cause DivisionByZero (max(1, total_optional_fields))."""
    assessment = make_assessment()
    result = compute_confidence(assessment, provided_fields_count=0, total_optional_fields=0)
    assert result["factors"]["coverage_factor"] == 0.75


# ==============================================================================
# 5. Limiting Reason Attribution Assertions
# ==============================================================================


@pytest.mark.parametrize("evidence_score", [0, 1, 2])
def test_limiting_reason_weak_evidence(evidence_score: int):
    """WEAK_OR_UNVERIFIED_EVIDENCE is triggered whenever evidence_score <= 2 (evidence_ceiling < 0.60)."""
    assessment = make_assessment(
        evidence_score=evidence_score,
        missing_information=[],
    )
    result = compute_confidence(assessment, provided_fields_count=3)
    assert result["factors"]["evidence_ceiling"] < 0.60
    assert result["limiting_reason"] == LIMITING_REASON_WEAK_EVIDENCE


def test_limiting_reason_precedence_weak_evidence_over_gaps():
    """Weak evidence (< 0.60) takes precedence over numerous information gaps (>= 0.15)."""
    all_gaps = list(MissingInfoCode)[:5]  # 5 items -> gap_penalty = 0.20 >= 0.15
    assessment = make_assessment(
        evidence_score=1,  # ceiling 0.44 < 0.60
        missing_information=all_gaps,
    )
    result = compute_confidence(assessment, provided_fields_count=3)
    assert result["factors"]["evidence_ceiling"] < 0.60
    assert result["factors"]["gap_penalty"] >= 0.15
    assert result["limiting_reason"] == LIMITING_REASON_WEAK_EVIDENCE


@pytest.mark.parametrize("gap_count", [4, 5, 6, 7])
def test_limiting_reason_numerous_information_gaps(gap_count: int):
    """NUMEROUS_INFORMATION_GAPS is triggered when evidence_score >= 3 and 4+ missing info codes exist."""
    codes = list(MissingInfoCode)[:gap_count]
    assessment = make_assessment(
        evidence_score=4,  # ceiling 0.86 >= 0.60
        missing_information=codes,
    )
    result = compute_confidence(assessment, provided_fields_count=3)
    assert result["factors"]["evidence_ceiling"] >= 0.60
    assert result["factors"]["gap_penalty"] >= 0.15
    assert result["limiting_reason"] == LIMITING_REASON_GAPS


@pytest.mark.parametrize("gap_count", [0, 1, 2, 3])
def test_limiting_reason_none_when_strong_evidence_and_few_gaps(gap_count: int):
    """limiting_reason is None when evidence score >= 3 (ceiling >= 0.60) and <= 3 gaps (penalty < 0.15)."""
    codes = list(MissingInfoCode)[:gap_count]
    assessment = make_assessment(
        evidence_score=3,  # ceiling 0.72 >= 0.60
        missing_information=codes,
    )
    result = compute_confidence(assessment, provided_fields_count=3)
    assert result["factors"]["evidence_ceiling"] >= 0.60
    assert result["factors"]["gap_penalty"] < 0.15
    assert result["limiting_reason"] is None


# ==============================================================================
# 6. Confidence Band Classification & Threshold Boundaries
# ==============================================================================


@pytest.mark.parametrize(
    "model_conf, evidence_score, fields, gaps, expected_value, expected_band",
    [
        # LOW: < 0.45
        (0.50, 1, 1, 2, 0.10, BAND_LOW),
        (0.60, 2, 3, 3, 0.23, BAND_LOW),
        (1.00, 0, 3, 0, 0.30, BAND_LOW),
        # MEDIUM: 0.45 <= value < 0.70
        (0.80, 3, 3, 2, 0.50, BAND_MEDIUM),
        (0.90, 4, 3, 2, 0.69, BAND_MEDIUM),
        # HIGH: >= 0.70
        (0.90, 4, 3, 1, 0.73, BAND_HIGH),
        (1.00, 5, 3, 0, 0.95, BAND_HIGH),
    ],
)
def test_confidence_band_classification(
    model_conf: float,
    evidence_score: int,
    fields: int,
    gaps: int,
    expected_value: float,
    expected_band: str,
):
    """Verify confidence values and band transitions across LOW, MEDIUM, and HIGH."""
    missing = list(MissingInfoCode)[:gaps]
    assessment = make_assessment(
        model_confidence=model_conf,
        evidence_score=evidence_score,
        missing_information=missing,
    )
    result = compute_confidence(assessment, provided_fields_count=fields, total_optional_fields=3)
    assert result["value"] == expected_value
    assert result["band"] == expected_band


def test_confidence_band_exact_transition_points():
    """Verify exact transition boundaries: 0.44 -> LOW, 0.45 -> MEDIUM, 0.69 -> MEDIUM, 0.70 -> HIGH."""
    # 1) Boundary around 0.45:
    # Let model_conf = 0.50, evidence_score = 5 (ceiling = 1.0), coverage = 1.0 (3/3 fields)
    # raw = 0.50 * 1.0 * 1.0 - gap_penalty
    # With gap_penalty = 0.05 (e.g. raw 0.45 -> MEDIUM)
    # With gap_penalty = 0.06 (raw 0.44 -> LOW)
    # We test using exact assessments:
    # 0.45: model_conf=0.65, evidence=3 (0.72), fields=3 (1.0), gaps=1 (0.04) -> 0.65 * 0.72 - 0.04 = 0.428 (LOW)
    # 0.45: model_conf=0.6806 * 0.72 - 0.04 = 0.450 -> MEDIUM
    assessment_low = make_assessment(
        model_confidence=0.60,
        evidence_score=3,  # ceiling 0.72
        missing_information=[],
    )
    # raw = 0.60 * 0.72 * 1.0 - 0.0 = 0.432 -> 0.43 -> LOW
    res_low = compute_confidence(assessment_low, provided_fields_count=3)
    assert res_low["value"] == 0.43
    assert res_low["band"] == BAND_LOW

    assessment_med = make_assessment(
        model_confidence=0.625,
        evidence_score=3,  # ceiling 0.72
        missing_information=[],
    )
    # raw = 0.625 * 0.72 * 1.0 - 0.0 = 0.45 -> 0.45 -> MEDIUM
    res_med = compute_confidence(assessment_med, provided_fields_count=3)
    assert res_med["value"] == 0.45
    assert res_med["band"] == BAND_MEDIUM

    assessment_high_boundary = make_assessment(
        model_confidence=0.70,
        evidence_score=5,  # ceiling 1.0
        missing_information=[],
    )
    # raw = 0.70 * 1.0 * 1.0 - 0.0 = 0.70 -> HIGH
    res_high = compute_confidence(assessment_high_boundary, provided_fields_count=3)
    assert res_high["value"] == 0.70
    assert res_high["band"] == BAND_HIGH


# ==============================================================================
# 7. Model Confidence Averaging (All 5 Dims + Severity)
# ==============================================================================


def test_base_model_confidence_equal_weighting_across_all_six_signals():
    """All 5 quality dimensions and severity confidence must be equally weighted in model_self_reported."""
    # 5 dimensions with 1.0 confidence, severity with 0.4 confidence
    # Mean = (5 * 1.0 + 0.4) / 6 = 5.4 / 6 = 0.90
    assessment = make_assessment(
        model_confidence=1.0,
        severity_confidence=0.4,
    )
    result = compute_confidence(assessment, provided_fields_count=3)
    assert result["factors"]["model_self_reported"] == 0.90


def test_single_dimension_confidence_impact():
    """Varying a single quality dimension equally influences model_self_reported."""
    # REPRODUCIBILITY has confidence 0.4, other 4 dims + severity have 1.0
    # Mean = (5 * 1.0 + 0.4) / 6 = 0.90
    assessment = make_assessment(
        model_confidence=1.0,
        dimension_confidences={QualityDimension.REPRODUCIBILITY: 0.4},
        severity_confidence=1.0,
    )
    result = compute_confidence(assessment, provided_fields_count=3)
    assert result["factors"]["model_self_reported"] == 0.90


# ==============================================================================
# 8. Return Contract Structure & Typing
# ==============================================================================


def test_return_contract_structure_and_factor_keys():
    """Ensure exact keys, types, and rounding in return structure."""
    assessment = make_assessment()
    result = compute_confidence(assessment, provided_fields_count=2, total_optional_fields=3)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"value", "band", "limiting_reason", "factors"}

    assert isinstance(result["value"], float)
    assert isinstance(result["band"], str)
    assert result["limiting_reason"] is None or isinstance(result["limiting_reason"], str)

    factors = result["factors"]
    assert isinstance(factors, dict)
    assert set(factors.keys()) == {
        "model_self_reported",
        "evidence_ceiling",
        "coverage_factor",
        "gap_penalty",
    }
    for factor_val in factors.values():
        assert isinstance(factor_val, float)
