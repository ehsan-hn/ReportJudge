from enum import Enum, IntEnum
import math

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


def test_rubric_versions():
    assert RUBRIC_VERSION == "1.0.0"
    assert PROMPT_VERSION == "v1_structured"


def test_quality_weights_sum():
    total = sum(QUALITY_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"Sum of weights must strictly equal 1.0, got {total}"


def test_all_quality_dimensions_in_weights():
    # Every QualityDimension member must be present in QUALITY_WEIGHTS
    for dimension in QualityDimension:
        assert dimension in QUALITY_WEIGHTS, f"Missing weight for dimension {dimension}"

    assert set(QUALITY_WEIGHTS.keys()) == set(QualityDimension)
    assert len(QUALITY_WEIGHTS) == 5

    # Specific weight allocations
    assert QUALITY_WEIGHTS[QualityDimension.CLARITY] == 0.20
    assert QUALITY_WEIGHTS[QualityDimension.EVIDENCE_STRENGTH] == 0.30
    assert QUALITY_WEIGHTS[QualityDimension.IMPACT_ARTICULATION] == 0.25
    assert QUALITY_WEIGHTS[QualityDimension.REPRODUCIBILITY] == 0.10
    assert QUALITY_WEIGHTS[QualityDimension.ACTION_CONTEXT] == 0.15


def test_quality_dimension_enum():
    assert issubclass(QualityDimension, (str, Enum))
    expected_members = [
        ("CLARITY", "CLARITY"),
        ("EVIDENCE_STRENGTH", "EVIDENCE_STRENGTH"),
        ("IMPACT_ARTICULATION", "IMPACT_ARTICULATION"),
        ("REPRODUCIBILITY", "REPRODUCIBILITY"),
        ("ACTION_CONTEXT", "ACTION_CONTEXT"),
    ]
    actual_members = [(member.name, member.value) for member in QualityDimension]
    assert actual_members == expected_members


def test_impact_scope_enum():
    assert issubclass(ImpactScope, IntEnum)
    expected_members = [
        ("NONE", 0),
        ("SINGLE_USER", 1),
        ("SMALL_SUBSET", 2),
        ("LARGE_SUBSET", 3),
        ("ALL_USERS", 4),
    ]
    actual_members = [(member.name, member.value) for member in ImpactScope]
    assert actual_members == expected_members

    # Ordering and comparisons
    assert ImpactScope.NONE < ImpactScope.SINGLE_USER < ImpactScope.SMALL_SUBSET < ImpactScope.LARGE_SUBSET < ImpactScope.ALL_USERS


def test_business_criticality_enum():
    assert issubclass(BusinessCriticality, IntEnum)
    expected_members = [
        ("COSMETIC", 0),
        ("DEGRADED_UX", 1),
        ("CORE_FLOW_IMPAIRED", 2),
        ("REVENUE_OR_DATA_AT_RISK", 3),
        ("DATA_LOSS_OR_BREACH", 4),
    ]
    actual_members = [(member.name, member.value) for member in BusinessCriticality]
    assert actual_members == expected_members

    # Ordering and comparisons
    assert (
        BusinessCriticality.COSMETIC
        < BusinessCriticality.DEGRADED_UX
        < BusinessCriticality.CORE_FLOW_IMPAIRED
        < BusinessCriticality.REVENUE_OR_DATA_AT_RISK
        < BusinessCriticality.DATA_LOSS_OR_BREACH
    )


def test_time_sensitivity_enum():
    assert issubclass(TimeSensitivity, IntEnum)
    expected_members = [
        ("STABLE", 0),
        ("SLOW_DEGRADATION", 1),
        ("ACTIVE_DEGRADATION", 2),
        ("RAPID_ESCALATION", 3),
    ]
    actual_members = [(member.name, member.value) for member in TimeSensitivity]
    assert actual_members == expected_members

    # Ordering and comparisons
    assert (
        TimeSensitivity.STABLE
        < TimeSensitivity.SLOW_DEGRADATION
        < TimeSensitivity.ACTIVE_DEGRADATION
        < TimeSensitivity.RAPID_ESCALATION
    )


def test_severity_level_enum():
    assert issubclass(SeverityLevel, (str, Enum))
    expected_members = [
        ("SEV1", "SEV1"),
        ("SEV2", "SEV2"),
        ("SEV3", "SEV3"),
        ("SEV4", "SEV4"),
    ]
    actual_members = [(member.name, member.value) for member in SeverityLevel]
    assert actual_members == expected_members


def test_missing_info_code_enum():
    assert issubclass(MissingInfoCode, (str, Enum))
    expected_members = [
        ("IMPACT_SCOPE_UNQUANTIFIED", "IMPACT_SCOPE_UNQUANTIFIED"),
        ("NO_TIMELINE", "NO_TIMELINE"),
        ("NO_ERROR_DETAILS", "NO_ERROR_DETAILS"),
        ("NO_REPRO_STEPS", "NO_REPRO_STEPS"),
        ("AFFECTED_COMPONENT_UNKNOWN", "AFFECTED_COMPONENT_UNKNOWN"),
        ("NO_MITIGATION_HISTORY", "NO_MITIGATION_HISTORY"),
        ("NO_MONITORING_DATA", "NO_MONITORING_DATA"),
        ("ENVIRONMENT_UNKNOWN", "ENVIRONMENT_UNKNOWN"),
    ]
    actual_members = [(member.name, member.value) for member in MissingInfoCode]
    assert actual_members == expected_members
