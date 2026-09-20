"""Unit and Schema Tests for Evaluation Benchmark Suite and Metrics.

Verifies:
1. Schema integrity, field completeness, and boundary criteria for eval/cases/golden.json.
2. Structure, property definitions, and paired cases for eval/cases/properties.json.
3. Correctness of metric calculation functions in eval/metrics.py under normal and edge conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from eval.metrics import (
    compute_floor_enforcement_rate,
    compute_invariance_pass_rate,
    compute_severity_accuracy,
    summarize_evaluation,
)

CASES_DIR = Path(__file__).resolve().parent.parent / "eval" / "cases"
GOLDEN_PATH = CASES_DIR / "golden.json"
PROPERTIES_PATH = CASES_DIR / "properties.json"


# ==============================================================================
# 1. Dataset Schema and Parsing Tests: golden.json
# ==============================================================================


def test_golden_cases_file_exists_and_parses():
    """golden.json exists, parses as valid JSON, and contains at least 6 cases."""
    assert GOLDEN_PATH.exists(), f"Missing {GOLDEN_PATH}"
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert isinstance(cases, list)
    assert len(cases) >= 6


def test_golden_cases_schema_and_required_keys():
    """Every case in golden.json contains required metadata, input, and expectation fields."""
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    case_ids = set()
    for case in cases:
        assert "id" in case and isinstance(case["id"], str)
        assert "name" in case and isinstance(case["name"], str)
        assert "input" in case and isinstance(case["input"], dict)
        assert "expected" in case and isinstance(case["expected"], dict)

        # Input fields
        inp = case["input"]
        assert "title" in inp and isinstance(inp["title"], str) and len(inp["title"]) > 0
        assert "description" in inp and isinstance(inp["description"], str) and len(inp["description"]) > 0
        assert "impact" in inp
        assert "evidence" in inp
        assert "actions_taken" in inp

        # Expected fields
        exp = case["expected"]
        assert "is_valid_incident_report" in exp and isinstance(exp["is_valid_incident_report"], bool)
        assert "must_require_human_review" in exp and isinstance(exp["must_require_human_review"], bool)

        case_ids.add(case["id"])

    required_ids = {
        "sev1_credential_leak",
        "sev2_total_outage_core_flow",
        "sev3_partial_degradation",
        "sev4_cosmetic_issue",
        "vague_ambiguous_report",
        "adversarial_injection",
    }
    assert required_ids.issubset(case_ids), f"Missing required IDs: {required_ids - case_ids}"


def test_golden_cases_specific_canonical_expectations():
    """Verify specific canonical incident assertions specified in task requirements."""
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        cases = {c["id"]: c for c in json.load(f)}

    # Case 1: sev1_credential_leak
    c1 = cases["sev1_credential_leak"]
    assert c1["expected"]["severity"] == "SEV1"
    assert c1["expected"]["expected_floor"] == "FLOOR_DATA_LOSS_OR_BREACH_SEV1"
    assert c1["expected"]["min_confidence"] == 0.70
    assert c1["expected"]["is_valid_incident_report"] is True
    assert c1["expected"]["must_require_human_review"] is True

    # Case 2: sev2_total_outage_core_flow
    c2 = cases["sev2_total_outage_core_flow"]
    assert c2["expected"]["severity"] == "SEV2"
    assert c2["expected"]["expected_floor"] == "FLOOR_TOTAL_OUTAGE_SEV2"
    assert c2["expected"]["is_valid_incident_report"] is True
    assert c2["expected"]["must_require_human_review"] is True

    # Case 3: sev3_partial_degradation
    c3 = cases["sev3_partial_degradation"]
    assert c3["expected"]["severity"] == "SEV3"
    assert c3["expected"]["is_valid_incident_report"] is True

    # Case 4: sev4_cosmetic_issue
    c4 = cases["sev4_cosmetic_issue"]
    assert c4["expected"]["severity"] == "SEV4"
    assert c4["expected"]["is_valid_incident_report"] is True

    # Case 5: vague_ambiguous_report
    c5 = cases["vague_ambiguous_report"]
    assert c5["expected"]["max_quality_score"] == 35.0
    assert c5["expected"]["max_confidence"] == 0.45
    assert c5["expected"]["must_require_human_review"] is True

    # Case 6: adversarial_injection
    c6 = cases["adversarial_injection"]
    assert c6["expected"]["is_valid_incident_report"] is False
    assert c6["expected"]["must_require_human_review"] is True


# ==============================================================================
# 2. Dataset Schema and Parsing Tests: properties.json
# ==============================================================================


def test_properties_file_exists_and_parses():
    """properties.json exists, parses as valid JSON, and contains 3 property suites."""
    assert PROPERTIES_PATH.exists(), f"Missing {PROPERTIES_PATH}"
    with open(PROPERTIES_PATH, "r", encoding="utf-8") as f:
        props = json.load(f)

    assert isinstance(props, list)
    assert len(props) >= 3


def test_properties_schema_and_canonical_properties():
    """properties.json defines required property invariants and paired test inputs."""
    with open(PROPERTIES_PATH, "r", encoding="utf-8") as f:
        props_list = json.load(f)

    props_by_id = {p["property_id"]: p for p in props_list}

    required_props = {
        "monotonicity_evidence",
        "monotonicity_field_coverage",
        "breach_policy_floor_invariance",
    }
    assert required_props.issubset(set(props_by_id.keys()))

    # Property 1: monotonicity_evidence
    p1 = props_by_id["monotonicity_evidence"]
    assert "cases" in p1
    assert "weak" in p1["cases"]
    assert "strong" in p1["cases"]
    assert p1["cases"]["weak"]["evidence"] is None
    assert p1["cases"]["strong"]["evidence"] is not None
    assert "datadog" in p1["cases"]["strong"]["evidence"].lower()
    assert "strong.confidence >= weak.confidence" in p1["assertion"]

    # Property 2: monotonicity_field_coverage
    p2 = props_by_id["monotonicity_field_coverage"]
    assert "cases" in p2
    assert "minimal" in p2["cases"]
    assert "complete" in p2["cases"]
    assert p2["cases"]["minimal"]["impact"] is None
    assert p2["cases"]["minimal"]["evidence"] is None
    assert p2["cases"]["minimal"]["actions_taken"] is None
    assert p2["cases"]["complete"]["impact"] is not None
    assert p2["cases"]["complete"]["evidence"] is not None
    assert p2["cases"]["complete"]["actions_taken"] is not None
    assert "complete.confidence >= minimal.confidence" in p2["assertion"]

    # Property 3: breach_policy_floor_invariance
    p3 = props_by_id["breach_policy_floor_invariance"]
    assert "cases" in p3
    assert p3.get("expected_severity") == "SEV1"
    assert p3.get("expected_floor") == "FLOOR_DATA_LOSS_OR_BREACH_SEV1"
    # Both variants must mention credentials / passwords / breach
    for variant in p3["cases"].values():
        content = (variant["title"] + " " + variant["description"]).lower()
        assert any(k in content for k in ["credential", "password", "breach"])


# ==============================================================================
# 3. Metric Function Tests: compute_severity_accuracy
# ==============================================================================


def test_compute_severity_accuracy_perfect_score():
    """Returns 1.0 when predictions and targets match exactly."""
    preds = ["SEV1", "SEV2", "SEV3", "SEV4"]
    tgts = ["SEV1", "SEV2", "SEV3", "SEV4"]
    assert compute_severity_accuracy(preds, tgts) == 1.0


def test_compute_severity_accuracy_partial_score():
    """Computes exact accuracy fraction for partial matches."""
    preds = ["SEV1", "SEV2", "SEV3", "SEV4"]
    tgts = ["SEV1", "SEV2", "SEV4", "SEV3"]
    assert compute_severity_accuracy(preds, tgts) == 0.5

    preds3 = ["SEV1", "SEV2", "SEV3"]
    tgts3 = ["SEV1", "SEV2", "SEV1"]
    assert compute_severity_accuracy(preds3, tgts3) == 0.6667


def test_compute_severity_accuracy_zero_score():
    """Returns 0.0 when no predictions match targets."""
    preds = ["SEV1", "SEV2"]
    tgts = ["SEV3", "SEV4"]
    assert compute_severity_accuracy(preds, tgts) == 0.0


def test_compute_severity_accuracy_empty_lists():
    """Returns 0.0 when input lists are empty."""
    assert compute_severity_accuracy([], []) == 0.0


def test_compute_severity_accuracy_mismatched_lengths():
    """Raises ValueError when predictions and targets lengths differ."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_severity_accuracy(["SEV1"], ["SEV1", "SEV2"])


def test_compute_severity_accuracy_normalization():
    """Handles lowercase strings and whitespace gracefully."""
    preds = [" sev1 ", "SEV2\n", "sev3"]
    tgts = ["SEV1", "sev2", "  SEV3  "]
    assert compute_severity_accuracy(preds, tgts) == 1.0


# ==============================================================================
# 4. Metric Function Tests: compute_floor_enforcement_rate
# ==============================================================================


def test_compute_floor_enforcement_rate_all_enforced():
    """Returns 1.0 when all expected floors are present in results."""
    results = [
        {"severity": {"applied_rules": ["FLOOR_DATA_LOSS_OR_BREACH_SEV1"]}},
        {"severity": {"applied_rules": ["FLOOR_TOTAL_OUTAGE_SEV2"]}},
        {"severity": {"applied_rules": []}},
    ]
    expected = ["FLOOR_DATA_LOSS_OR_BREACH_SEV1", "FLOOR_TOTAL_OUTAGE_SEV2", None]
    assert compute_floor_enforcement_rate(results, expected) == 1.0


def test_compute_floor_enforcement_rate_partial_enforced():
    """Returns 0.5 when only 1 out of 2 expected floors is enforced."""
    results = [
        {"severity": {"applied_rules": ["FLOOR_DATA_LOSS_OR_BREACH_SEV1"]}},
        {"severity": {"applied_rules": []}},  # Missed expected floor
        {"severity": {"applied_rules": []}},
    ]
    expected = ["FLOOR_DATA_LOSS_OR_BREACH_SEV1", "FLOOR_TOTAL_OUTAGE_SEV2", None]
    assert compute_floor_enforcement_rate(results, expected) == 0.5


def test_compute_floor_enforcement_rate_zero_enforced():
    """Returns 0.0 when expected floors are missing from results."""
    results = [
        {"severity": {"applied_rules": []}},
        {"severity": {"applied_rules": []}},
    ]
    expected = ["FLOOR_DATA_LOSS_OR_BREACH_SEV1", "FLOOR_TOTAL_OUTAGE_SEV2"]
    assert compute_floor_enforcement_rate(results, expected) == 0.0


def test_compute_floor_enforcement_rate_no_floors_expected():
    """Returns 1.0 when no floors are expected and none are applied."""
    results = [
        {"severity": {"applied_rules": []}},
        {"severity": {"applied_rules": []}},
    ]
    expected = [None, None]
    assert compute_floor_enforcement_rate(results, expected) == 1.0


def test_compute_floor_enforcement_rate_strict_mode():
    """strict=True verifies both positive presence and negative absence."""
    # Case 1 matches floor, Case 2 matches None (empty)
    results = [
        {"severity": {"applied_rules": ["FLOOR_DATA_LOSS_OR_BREACH_SEV1"]}},
        {"severity": {"applied_rules": []}},
    ]
    expected = ["FLOOR_DATA_LOSS_OR_BREACH_SEV1", None]
    assert compute_floor_enforcement_rate(results, expected, strict=True) == 1.0

    # Case 2 has unexpected floor -> strict match fails for Case 2
    results_err = [
        {"severity": {"applied_rules": ["FLOOR_DATA_LOSS_OR_BREACH_SEV1"]}},
        {"severity": {"applied_rules": ["UNEXPECTED_FLOOR"]}},
    ]
    assert compute_floor_enforcement_rate(results_err, expected, strict=True) == 0.5


def test_compute_floor_enforcement_rate_empty_and_mismatch():
    """Handles empty list gracefully and raises ValueError on mismatched lengths."""
    assert compute_floor_enforcement_rate([], []) == 0.0

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_floor_enforcement_rate([{}], [None, None])


# ==============================================================================
# 5. Metric Function Tests: compute_invariance_pass_rate
# ==============================================================================


def test_compute_invariance_pass_rate_all_pass():
    """Returns 1.0 when all invariance assertions pass."""
    assert compute_invariance_pass_rate([True, True, True]) == 1.0


def test_compute_invariance_pass_rate_mixed():
    """Computes correct fraction of passed invariance tests."""
    assert compute_invariance_pass_rate([True, False, True, False]) == 0.5
    assert compute_invariance_pass_rate([True, True, False]) == 0.6667


def test_compute_invariance_pass_rate_all_fail():
    """Returns 0.0 when all assertions fail."""
    assert compute_invariance_pass_rate([False, False]) == 0.0


def test_compute_invariance_pass_rate_empty():
    """Returns 0.0 on empty input."""
    assert compute_invariance_pass_rate([]) == 0.0


# ==============================================================================
# 6. Summary Metric Tests: summarize_evaluation
# ==============================================================================


def test_summarize_evaluation_empty():
    """Empty results list returns zeroed statistics structure."""
    summary = summarize_evaluation([])
    assert summary["total_cases"] == 0
    assert summary["mean_quality_score"] == 0.0
    assert summary["mean_confidence"] == 0.0
    assert summary["human_review_flag_rate"] == 0.0
    assert isinstance(summary["band_distribution"], dict)
    assert summary["band_distribution"]["STRONG"] == 0
    assert summary["band_distribution"]["LOW"] == 0


def test_summarize_evaluation_standard():
    """Aggregates mean quality score, confidence, bands, and human review rates."""
    results = [
        {
            "report_quality": {"score": 80.0, "band": "STRONG"},
            "confidence": {"value": 0.85, "band": "HIGH"},
            "requires_human_review": False,
        },
        {
            "report_quality": {"score": 60.0, "band": "ADEQUATE"},
            "confidence": {"value": 0.65, "band": "MEDIUM"},
            "requires_human_review": True,
        },
        {
            "report_quality": {"score": 20.0, "band": "POOR"},
            "confidence": {"value": 0.30, "band": "LOW"},
            "requires_human_review": True,
        },
    ]

    summary = summarize_evaluation(results)

    assert summary["total_cases"] == 3
    # Mean quality score = (80.0 + 60.0 + 20.0) / 3 = 53.33
    assert summary["mean_quality_score"] == 53.33
    # Mean confidence = (0.85 + 0.65 + 0.30) / 3 = 0.60
    assert summary["mean_confidence"] == 0.60
    # Human review flag rate = 2 / 3 = 0.6667
    assert summary["human_review_flag_rate"] == 0.6667

    # Quality bands
    assert summary["quality_band_distribution"]["STRONG"] == 1
    assert summary["quality_band_distribution"]["ADEQUATE"] == 1
    assert summary["quality_band_distribution"]["POOR"] == 1
    assert summary["quality_band_distribution"]["WEAK"] == 0

    # Confidence bands
    assert summary["confidence_band_distribution"]["HIGH"] == 1
    assert summary["confidence_band_distribution"]["MEDIUM"] == 1
    assert summary["confidence_band_distribution"]["LOW"] == 1

    # Combined band distribution contains both
    assert summary["band_distribution"]["STRONG"] == 1
    assert summary["band_distribution"]["LOW"] == 1


def test_summarize_evaluation_flat_and_duck_typed_structures():
    """Gracefully extracts metrics from flat dictionaries as well."""
    results = [
        {
            "quality_score": 90.0,
            "quality_band": "STRONG",
            "confidence_value": 0.90,
            "confidence_band": "HIGH",
            "must_require_human_review": False,
        },
        {
            "quality_score": 50.0,
            "quality_band": "WEAK",
            "confidence_value": 0.50,
            "confidence_band": "MEDIUM",
            "must_require_human_review": True,
        },
    ]

    summary = summarize_evaluation(results)

    assert summary["total_cases"] == 2
    assert summary["mean_quality_score"] == 70.0
    assert summary["mean_confidence"] == 0.70
    assert summary["human_review_flag_rate"] == 0.5
    assert summary["quality_band_distribution"]["STRONG"] == 1
    assert summary["quality_band_distribution"]["WEAK"] == 1
