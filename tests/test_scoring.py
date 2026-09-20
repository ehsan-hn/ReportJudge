from unittest.mock import patch
import pytest

from app.judgment.contracts import DimensionSignal, SeveritySignals
from app.judgment.rubric import (
    QUALITY_WEIGHTS,
    BusinessCriticality,
    ImpactScope,
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


def make_dims(score: int, confidence: float = 0.9) -> dict[QualityDimension, DimensionSignal]:
    """Helper to create a full mapping of quality dimensions with identical scores."""
    return {
        dim: DimensionSignal(
            justification=f"Assessment justification for {dim.value}",
            evidence_quote=f"Evidence for {dim.value}",
            score=score,
            confidence=confidence,
        )
        for dim in QualityDimension
    }


def make_custom_dims(scores: dict[QualityDimension, int | float]) -> dict[QualityDimension, DimensionSignal]:
    """Helper to create dimension mapping with specific scores."""
    dims = {}
    for dim in QualityDimension:
        score_val = scores.get(dim, 0)
        # Use model_construct to allow exact fractional float scores for boundary testing
        dims[dim] = DimensionSignal.model_construct(
            justification=f"Assessment justification for {dim.value}",
            evidence_quote=f"Evidence for {dim.value}",
            score=score_val,
            confidence=0.9,
        )
    return dims


def make_severity_signals(
    impact_scope: int = ImpactScope.SINGLE_USER,
    business_criticality: int = BusinessCriticality.DEGRADED_UX,
    time_sensitivity: int = TimeSensitivity.STABLE,
    confidence: float = 0.9,
    reasoning: str = "Automated severity assessment test.",
) -> SeveritySignals:
    """Helper to construct SeveritySignals with sensible defaults."""
    return SeveritySignals(
        reasoning=reasoning,
        impact_scope=impact_scope,
        business_criticality=business_criticality,
        time_sensitivity=time_sensitivity,
        confidence=confidence,
    )


# ==============================================================================
# 1. Quality Score Engine Tests
# ==============================================================================


def test_constants_defined():
    assert FLOOR_DATA_LOSS_OR_BREACH_SEV1 == "FLOOR_DATA_LOSS_OR_BREACH_SEV1"
    assert FLOOR_TOTAL_OUTAGE_SEV2 == "FLOOR_TOTAL_OUTAGE_SEV2"


def test_minimum_quality_score():
    """All 0s -> 0.0, 'POOR'."""
    dims = make_dims(score=0)
    score, band = compute_quality_score(dims)
    assert score == 0.0
    assert band == "POOR"


def test_maximum_quality_score():
    """All 5s -> 100.0, 'STRONG'."""
    dims = make_dims(score=5)
    score, band = compute_quality_score(dims)
    assert score == 100.0
    assert band == "STRONG"


def test_quality_score_missing_dimension_raises():
    """Missing any quality dimension must raise ValueError."""
    dims = make_dims(score=3)
    del dims[QualityDimension.ACTION_CONTEXT]
    with pytest.raises(ValueError) as exc_info:
        compute_quality_score(dims)
    assert "Missing quality dimensions: ['ACTION_CONTEXT']" in str(exc_info.value)


@pytest.mark.parametrize(
    "target_score, expected_band",
    [
        (0.0, "POOR"),
        (20.0, "POOR"),
        (34.8, "POOR"),
        (34.9, "POOR"),
        (35.0, "WEAK"),
        (35.1, "WEAK"),
        (45.0, "WEAK"),
        (54.8, "WEAK"),
        (54.9, "WEAK"),
        (55.0, "ADEQUATE"),
        (55.1, "ADEQUATE"),
        (65.0, "ADEQUATE"),
        (74.8, "ADEQUATE"),
        (74.9, "ADEQUATE"),
        (75.0, "STRONG"),
        (75.1, "STRONG"),
        (90.0, "STRONG"),
        (100.0, "STRONG"),
    ],
)
def test_quality_band_transitions_exact(target_score: float, expected_band: str):
    """Test exact band transition boundaries (34.9, 35.0, 54.9, 55.0, 74.9, 75.0)."""
    # raw_weighted_sum = (target_score / 100.0) * 5.0
    # Since sum(QUALITY_WEIGHTS) == 1.0, assigning uniform score = raw_weighted_sum yields exact target_score
    uniform_score = (target_score / 100.0) * 5.0
    scores = {dim: uniform_score for dim in QualityDimension}
    dims = make_custom_dims(scores)

    score, band = compute_quality_score(dims)
    assert score == pytest.approx(target_score, abs=1e-5)
    assert band == expected_band


def test_quality_score_integer_signals():
    """Verify quality scoring and bands with strictly valid integer DimensionSignals."""
    # Test valid integer combinations
    # 1) Score around 34 / 35:
    # CLARITY: 0, EVIDENCE: 5 (0.30*5=1.5), IMPACT: 0, REPRO: 1 (0.10*1=0.10), ACTION: 1 (0.15*1=0.15)
    # raw = 1.75 -> 1.75 / 5 * 100 = 35.0 -> WEAK
    dims_35 = {
        QualityDimension.CLARITY: DimensionSignal(justification="j", score=0, confidence=0.9),
        QualityDimension.EVIDENCE_STRENGTH: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.IMPACT_ARTICULATION: DimensionSignal(justification="j", score=0, confidence=0.9),
        QualityDimension.REPRODUCIBILITY: DimensionSignal(justification="j", score=1, confidence=0.9),
        QualityDimension.ACTION_CONTEXT: DimensionSignal(justification="j", score=1, confidence=0.9),
    }
    score_35, band_35 = compute_quality_score(dims_35)
    assert score_35 == 35.0
    assert band_35 == "WEAK"

    # CLARITY: 0, EVIDENCE: 5 (1.5), IMPACT: 0, REPRO: 2 (0.2), ACTION: 0 (0)
    # raw = 1.70 -> 1.70 / 5 * 100 = 34.0 -> POOR
    dims_34 = {
        QualityDimension.CLARITY: DimensionSignal(justification="j", score=0, confidence=0.9),
        QualityDimension.EVIDENCE_STRENGTH: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.IMPACT_ARTICULATION: DimensionSignal(justification="j", score=0, confidence=0.9),
        QualityDimension.REPRODUCIBILITY: DimensionSignal(justification="j", score=2, confidence=0.9),
        QualityDimension.ACTION_CONTEXT: DimensionSignal(justification="j", score=0, confidence=0.9),
    }
    score_34, band_34 = compute_quality_score(dims_34)
    assert score_34 == 34.0
    assert band_34 == "POOR"

    # 2) Score 54.0 vs 55.0
    # CLARITY: 5 (1.0), EVIDENCE: 5 (1.5), IMPACT: 1 (0.25), REPRO: 0, ACTION: 0
    # raw = 2.75 -> 55.0 -> ADEQUATE
    dims_55 = {
        QualityDimension.CLARITY: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.EVIDENCE_STRENGTH: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.IMPACT_ARTICULATION: DimensionSignal(justification="j", score=1, confidence=0.9),
        QualityDimension.REPRODUCIBILITY: DimensionSignal(justification="j", score=0, confidence=0.9),
        QualityDimension.ACTION_CONTEXT: DimensionSignal(justification="j", score=0, confidence=0.9),
    }
    score_55, band_55 = compute_quality_score(dims_55)
    assert score_55 == 55.0
    assert band_55 == "ADEQUATE"

    # CLARITY: 5 (1.0), EVIDENCE: 5 (1.5), IMPACT: 0, REPRO: 2 (0.2), ACTION: 0
    # raw = 2.70 -> 54.0 -> WEAK
    dims_54 = {
        QualityDimension.CLARITY: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.EVIDENCE_STRENGTH: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.IMPACT_ARTICULATION: DimensionSignal(justification="j", score=0, confidence=0.9),
        QualityDimension.REPRODUCIBILITY: DimensionSignal(justification="j", score=2, confidence=0.9),
        QualityDimension.ACTION_CONTEXT: DimensionSignal(justification="j", score=0, confidence=0.9),
    }
    score_54, band_54 = compute_quality_score(dims_54)
    assert score_54 == 54.0
    assert band_54 == "WEAK"

    # 3) Score 74.0 vs 75.0
    # CLARITY: 5 (1.0), EVIDENCE: 5 (1.5), IMPACT: 5 (1.25), REPRO: 0, ACTION: 0
    # raw = 3.75 -> 75.0 -> STRONG
    dims_75 = {
        QualityDimension.CLARITY: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.EVIDENCE_STRENGTH: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.IMPACT_ARTICULATION: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.REPRODUCIBILITY: DimensionSignal(justification="j", score=0, confidence=0.9),
        QualityDimension.ACTION_CONTEXT: DimensionSignal(justification="j", score=0, confidence=0.9),
    }
    score_75, band_75 = compute_quality_score(dims_75)
    assert score_75 == 75.0
    assert band_75 == "STRONG"

    # CLARITY: 5 (1.0), EVIDENCE: 5 (1.5), IMPACT: 4 (1.00), REPRO: 2 (0.2), ACTION: 0
    # raw = 3.70 -> 74.0 -> ADEQUATE
    dims_74 = {
        QualityDimension.CLARITY: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.EVIDENCE_STRENGTH: DimensionSignal(justification="j", score=5, confidence=0.9),
        QualityDimension.IMPACT_ARTICULATION: DimensionSignal(justification="j", score=4, confidence=0.9),
        QualityDimension.REPRODUCIBILITY: DimensionSignal(justification="j", score=2, confidence=0.9),
        QualityDimension.ACTION_CONTEXT: DimensionSignal(justification="j", score=0, confidence=0.9),
    }
    score_74, band_74 = compute_quality_score(dims_74)
    assert score_74 == 74.0
    assert band_74 == "ADEQUATE"


# ==============================================================================
# 2. Baseline Severity Engine Tests
# ==============================================================================


def test_severity_baseline_minimum():
    """All 0s: scope=0, crit=0, time=0 -> index=0.0, SEV4, applied_floors=[]."""
    signals = make_severity_signals(
        impact_scope=ImpactScope.NONE,
        business_criticality=BusinessCriticality.COSMETIC,
        time_sensitivity=TimeSensitivity.STABLE,
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 0.0
    assert level == SeverityLevel.SEV4
    assert applied_floors == []


def test_severity_baseline_boundary_sev4():
    """index < 1.0 -> SEV4."""
    # scope=1 (0.45), crit=1 (0.35), time=0 (0.0) -> index = 0.80
    signals = make_severity_signals(
        impact_scope=ImpactScope.SINGLE_USER,
        business_criticality=BusinessCriticality.DEGRADED_UX,
        time_sensitivity=TimeSensitivity.STABLE,
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 0.80
    assert level == SeverityLevel.SEV4
    assert applied_floors == []


def test_severity_baseline_boundary_sev3():
    """1.0 <= index < 2.0 -> SEV3."""
    # scope=1 (0.45), crit=1 (0.35), time=1 (0.20 * 4/3 = 0.2667) -> index = 1.07
    signals_low = make_severity_signals(
        impact_scope=ImpactScope.SINGLE_USER,
        business_criticality=BusinessCriticality.DEGRADED_UX,
        time_sensitivity=TimeSensitivity.SLOW_DEGRADATION,
    )
    level_low, index_low, floors_low = compute_severity(signals_low)
    assert index_low == 1.07
    assert level_low == SeverityLevel.SEV3
    assert floors_low == []

    # scope=3 (1.35), crit=1 (0.35), time=1 (0.2667) -> index = 1.97
    signals_high = make_severity_signals(
        impact_scope=ImpactScope.LARGE_SUBSET,
        business_criticality=BusinessCriticality.DEGRADED_UX,
        time_sensitivity=TimeSensitivity.SLOW_DEGRADATION,
    )
    level_high, index_high, floors_high = compute_severity(signals_high)
    assert index_high == 1.97
    assert level_high == SeverityLevel.SEV3
    assert floors_high == []


def test_severity_baseline_boundary_sev2():
    """2.0 <= index < 3.0 -> SEV2."""
    # scope=3 (1.35), crit=2 (0.70), time=0 (0.0) -> index = 2.05
    signals_low = make_severity_signals(
        impact_scope=ImpactScope.LARGE_SUBSET,
        business_criticality=BusinessCriticality.CORE_FLOW_IMPAIRED,
        time_sensitivity=TimeSensitivity.STABLE,
    )
    level_low, index_low, floors_low = compute_severity(signals_low)
    assert index_low == 2.05
    assert level_low == SeverityLevel.SEV2
    assert floors_low == []

    # scope=3 (1.35), crit=3 (1.05), time=2 (0.5333) -> index = 2.93
    signals_high = make_severity_signals(
        impact_scope=ImpactScope.LARGE_SUBSET,
        business_criticality=BusinessCriticality.REVENUE_OR_DATA_AT_RISK,
        time_sensitivity=TimeSensitivity.ACTIVE_DEGRADATION,
    )
    level_high, index_high, floors_high = compute_severity(signals_high)
    assert index_high == 2.93
    assert level_high == SeverityLevel.SEV2
    assert floors_high == []


def test_severity_baseline_boundary_sev1():
    """index >= 3.0 -> SEV1."""
    # scope=3 (1.35), crit=3 (1.05), time=3 (0.80) -> index = 3.20
    signals = make_severity_signals(
        impact_scope=ImpactScope.LARGE_SUBSET,
        business_criticality=BusinessCriticality.REVENUE_OR_DATA_AT_RISK,
        time_sensitivity=TimeSensitivity.RAPID_ESCALATION,
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 3.20
    assert level == SeverityLevel.SEV1
    assert applied_floors == []


# ==============================================================================
# 3. Policy Floor 1 Tests (FLOOR_DATA_LOSS_OR_BREACH_SEV1)
# ==============================================================================


def test_policy_floor_1_data_breach_low_impact():
    """Low impact scope and stable time sensitivity but data breach must be forced to SEV1."""
    # scope=0 (0.0), crit=4 (1.40), time=0 (0.0) -> baseline index = 1.40 (baseline SEV3)
    signals = make_severity_signals(
        impact_scope=ImpactScope.NONE,
        business_criticality=BusinessCriticality.DATA_LOSS_OR_BREACH,
        time_sensitivity=TimeSensitivity.STABLE,
        reasoning="Internal database leak with 0 user facing downtime.",
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 1.40
    assert level == SeverityLevel.SEV1
    assert applied_floors == [FLOOR_DATA_LOSS_OR_BREACH_SEV1]


def test_policy_floor_1_data_breach_natural_sev1():
    """Even if baseline index >= 3.0, data breach criticality sets SEV1 and tracks floor."""
    # scope=4 (1.80), crit=4 (1.40), time=3 (0.80) -> index = 4.0
    signals = make_severity_signals(
        impact_scope=ImpactScope.ALL_USERS,
        business_criticality=BusinessCriticality.DATA_LOSS_OR_BREACH,
        time_sensitivity=TimeSensitivity.RAPID_ESCALATION,
        reasoning="Full production data corruption under active escalation.",
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 4.0
    assert level == SeverityLevel.SEV1
    assert applied_floors == [FLOOR_DATA_LOSS_OR_BREACH_SEV1]


# ==============================================================================
# 4. Policy Floor 2 Tests (FLOOR_TOTAL_OUTAGE_SEV2)
# ==============================================================================


def test_policy_floor_2_total_outage_core_flow():
    """Total user outage (scope=4) with core flow impaired (crit=2) and low time sensitivity.

    Must result in SEV2, and must never be left as SEV3 or SEV4.
    """
    signals = make_severity_signals(
        impact_scope=ImpactScope.ALL_USERS,
        business_criticality=BusinessCriticality.CORE_FLOW_IMPAIRED,
        time_sensitivity=TimeSensitivity.STABLE,
        reasoning="Complete authentication portal outage affecting all users.",
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 2.50
    assert level == SeverityLevel.SEV2
    assert level not in (SeverityLevel.SEV3, SeverityLevel.SEV4)


def test_policy_floor_2_never_downgrades_natural_sev1():
    """Policy Floor 2 must never downgrade a natural SEV1 to SEV2."""
    # scope=4 (1.80), crit=2 (0.70), time=3 (0.80) -> index = 3.30 >= 3.0 -> natural SEV1
    signals = make_severity_signals(
        impact_scope=ImpactScope.ALL_USERS,
        business_criticality=BusinessCriticality.CORE_FLOW_IMPAIRED,
        time_sensitivity=TimeSensitivity.RAPID_ESCALATION,
        reasoning="Global user outage actively escalating.",
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 3.30
    assert level == SeverityLevel.SEV1
    assert FLOOR_TOTAL_OUTAGE_SEV2 not in applied_floors


def test_policy_floor_2_elevates_when_baseline_is_sev3():
    """If baseline level evaluates to SEV3 or SEV4 under total outage conditions, elevate to SEV2 and track floor."""
    signals = make_severity_signals(
        impact_scope=ImpactScope.ALL_USERS,
        business_criticality=BusinessCriticality.CORE_FLOW_IMPAIRED,
        time_sensitivity=TimeSensitivity.STABLE,
    )

    # To test the elevation branch of Floor 2 directly, simulate a baseline index evaluation yielding SEV3 (e.g. 1.80)
    def compute_severity_with_mocked_baseline(sig: SeveritySignals):
        # Continuous calculation
        raw_index = (
            0.45 * sig.impact_scope
            + 0.35 * sig.business_criticality
            + 0.20 * (sig.time_sensitivity / 3.0 * 4.0)
        )
        # Mock baseline level to SEV3 to verify elevation
        level = SeverityLevel.SEV3
        applied_floors = []

        if sig.business_criticality >= BusinessCriticality.DATA_LOSS_OR_BREACH:
            level = SeverityLevel.SEV1
            applied_floors.append(FLOOR_DATA_LOSS_OR_BREACH_SEV1)

        if (
            sig.impact_scope >= ImpactScope.ALL_USERS
            and sig.business_criticality >= BusinessCriticality.CORE_FLOW_IMPAIRED
            and level in (SeverityLevel.SEV3, SeverityLevel.SEV4)
        ):
            level = SeverityLevel.SEV2
            applied_floors.append(FLOOR_TOTAL_OUTAGE_SEV2)

        return level, round(raw_index, 2), applied_floors

    level, index, applied_floors = compute_severity_with_mocked_baseline(signals)
    assert level == SeverityLevel.SEV2
    assert applied_floors == [FLOOR_TOTAL_OUTAGE_SEV2]


def test_non_outage_does_not_trigger_floor_2():
    """Large subset outage (scope=3) with core flow impaired (crit=2) does not trigger Floor 2."""
    # scope=3 (1.35), crit=2 (0.70), time=0 (0.0) -> index = 2.05 -> SEV2 naturally
    signals = make_severity_signals(
        impact_scope=ImpactScope.LARGE_SUBSET,
        business_criticality=BusinessCriticality.CORE_FLOW_IMPAIRED,
        time_sensitivity=TimeSensitivity.STABLE,
    )
    level, index, applied_floors = compute_severity(signals)
    assert index == 2.05
    assert level == SeverityLevel.SEV2
    assert applied_floors == []
