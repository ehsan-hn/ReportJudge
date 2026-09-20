"""Evaluation Metrics and Benchmark Aggregators.

Provides quantitative evaluation metrics for incident scoring accuracy,
deterministic policy floor enforcement rates, property invariance pass rates,
and aggregate benchmark distribution statistics.
"""

from __future__ import annotations

import statistics
from typing import Any


def _normalize_severity(val: Any) -> str:
    """Normalize severity string/enum representation."""
    if hasattr(val, "value"):
        return str(val.value).strip().upper()
    return str(val).strip().upper()


def compute_severity_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Compute overall severity classification accuracy.

    Args:
        predictions: Predicted severity levels (e.g. ['SEV1', 'SEV2']).
        targets: Target/ground-truth severity levels.

    Returns:
        float: Fraction of exact severity matches in range [0.0, 1.0].

    Raises:
        ValueError: If predictions and targets have different lengths.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: predictions has {len(predictions)} items, "
            f"targets has {len(targets)} items."
        )

    if not predictions:
        return 0.0

    matches = sum(
        1
        for pred, tgt in zip(predictions, targets)
        if _normalize_severity(pred) == _normalize_severity(tgt)
    )
    return round(matches / len(predictions), 4)


def _extract_applied_floors(result: Any) -> list[str]:
    """Extract applied policy floor identifiers from an assessment result."""
    if isinstance(result, dict):
        severity = result.get("severity")
        if isinstance(severity, dict):
            rules = severity.get("applied_rules") or severity.get("applied_floors")
            if rules is not None:
                return list(rules)
        elif hasattr(severity, "applied_rules"):
            return list(getattr(severity, "applied_rules", []))
        elif hasattr(severity, "applied_floors"):
            return list(getattr(severity, "applied_floors", []))

        rules = result.get("applied_rules") or result.get("applied_floors")
        if rules is not None:
            return list(rules)

    elif hasattr(result, "severity"):
        severity = getattr(result, "severity")
        if isinstance(severity, dict):
            rules = severity.get("applied_rules") or severity.get("applied_floors")
            if rules is not None:
                return list(rules)
        elif hasattr(severity, "applied_rules"):
            return list(getattr(severity, "applied_rules", []))
        elif hasattr(severity, "applied_floors"):
            return list(getattr(severity, "applied_floors", []))

    elif hasattr(result, "applied_rules"):
        return list(getattr(result, "applied_rules", []))

    return []


def compute_floor_enforcement_rate(
    results: list[dict],
    expected_floors: list[str | None],
    *,
    strict: bool = False,
) -> float:
    """Compute the policy floor enforcement rate across evaluated results.

    By default (strict=False), measures the recall/enforcement rate of required
    policy floors (i.e. of all cases where expected_floor is not None, in what fraction
    was the required policy floor applied). If no cases specify an expected floor,
    returns 1.0 if no unexpected floors were applied, or 0.0 if empty.

    When strict=True, every case is checked: expected floors must be applied, and
    cases where expected_floor is None must have zero applied floors.

    Args:
        results: Evaluated assessment result dicts or domain objects.
        expected_floors: List of expected policy floor identifiers (or None if no floor expected).
        strict: If True, requires exact floor alignment across all cases (including None).

    Returns:
        float: Enforcement rate in range [0.0, 1.0].

    Raises:
        ValueError: If results and expected_floors have different lengths.
    """
    if len(results) != len(expected_floors):
        raise ValueError(
            f"Length mismatch: results has {len(results)} items, "
            f"expected_floors has {len(expected_floors)} items."
        )

    if not results:
        return 0.0

    if strict:
        matches = 0
        for res, exp in zip(results, expected_floors):
            applied = _extract_applied_floors(res)
            if exp is not None:
                if exp in applied:
                    matches += 1
            else:
                if len(applied) == 0:
                    matches += 1
        return round(matches / len(results), 4)

    # Standard enforcement rate (recall of required floors)
    cases_with_expected_floor = [
        (res, exp)
        for res, exp in zip(results, expected_floors)
        if exp is not None
    ]

    if not cases_with_expected_floor:
        # If no policy floors were expected across any case, verify no floors were falsely applied
        erroneous_floors = sum(
            1 for res in results if len(_extract_applied_floors(res)) > 0
        )
        return 1.0 if erroneous_floors == 0 else 0.0

    enforced = sum(
        1
        for res, exp in cases_with_expected_floor
        if exp in _extract_applied_floors(res)
    )
    return round(enforced / len(cases_with_expected_floor), 4)


def compute_invariance_pass_rate(invariance_results: list[bool]) -> float:
    """Compute the pass rate across invariant property test pairs.

    Args:
        invariance_results: List of booleans indicating whether each invariant held.

    Returns:
        float: Fraction of passed invariance tests in range [0.0, 1.0].
    """
    if not invariance_results:
        return 0.0

    passed = sum(1 for passed_test in invariance_results if bool(passed_test))
    return round(passed / len(invariance_results), 4)


def _extract_quality_score(res: Any) -> float | None:
    """Extract numeric quality score from result."""
    if isinstance(res, dict):
        rq = res.get("report_quality")
        if isinstance(rq, dict) and "score" in rq:
            return float(rq["score"])
        elif hasattr(rq, "score"):
            return float(getattr(rq, "score"))
        for k in ("quality_score", "score"):
            if k in res and res[k] is not None:
                return float(res[k])
    elif hasattr(res, "report_quality"):
        rq = getattr(res, "report_quality")
        if isinstance(rq, dict) and "score" in rq:
            return float(rq["score"])
        elif hasattr(rq, "score"):
            return float(getattr(rq, "score"))
    return None


def _extract_confidence_value(res: Any) -> float | None:
    """Extract numeric confidence value from result."""
    if isinstance(res, dict):
        conf = res.get("confidence")
        if isinstance(conf, dict) and "value" in conf:
            return float(conf["value"])
        elif hasattr(conf, "value"):
            return float(getattr(conf, "value"))
        elif isinstance(conf, (int, float)):
            return float(conf)
        for k in ("confidence_value", "confidence_score"):
            if k in res and res[k] is not None:
                return float(res[k])
    elif hasattr(res, "confidence"):
        conf = getattr(res, "confidence")
        if isinstance(conf, dict) and "value" in conf:
            return float(conf["value"])
        elif hasattr(conf, "value"):
            return float(getattr(conf, "value"))
        elif isinstance(conf, (int, float)):
            return float(conf)
    return None


def _extract_quality_band(res: Any) -> str | None:
    """Extract quality band string from result."""
    if isinstance(res, dict):
        rq = res.get("report_quality")
        if isinstance(rq, dict) and "band" in rq:
            return str(rq["band"]).upper()
        elif hasattr(rq, "band"):
            return str(getattr(rq, "band")).upper()
        for k in ("quality_band", "band"):
            if k in res and res[k] is not None:
                return str(res[k]).upper()
    elif hasattr(res, "report_quality"):
        rq = getattr(res, "report_quality")
        if isinstance(rq, dict) and "band" in rq:
            return str(rq["band"]).upper()
        elif hasattr(rq, "band"):
            return str(getattr(rq, "band")).upper()
    return None


def _extract_confidence_band(res: Any) -> str | None:
    """Extract confidence band string from result."""
    if isinstance(res, dict):
        conf = res.get("confidence")
        if isinstance(conf, dict) and "band" in conf:
            return str(conf["band"]).upper()
        elif hasattr(conf, "band"):
            return str(getattr(conf, "band")).upper()
        for k in ("confidence_band",):
            if k in res and res[k] is not None:
                return str(res[k]).upper()
    elif hasattr(res, "confidence"):
        conf = getattr(res, "confidence")
        if isinstance(conf, dict) and "band" in conf:
            return str(conf["band"]).upper()
        elif hasattr(conf, "band"):
            return str(getattr(conf, "band")).upper()
    return None


def _extract_human_review(res: Any) -> bool:
    """Extract human review requirement flag from result."""
    if isinstance(res, dict):
        for k in ("requires_human_review", "must_require_human_review", "human_review"):
            if k in res and res[k] is not None:
                return bool(res[k])
    elif hasattr(res, "requires_human_review"):
        return bool(getattr(res, "requires_human_review"))
    return False


def summarize_evaluation(results: list[dict]) -> dict[str, Any]:
    """Summarize an evaluation run with aggregate statistics.

    Computes:
    - mean quality score
    - mean confidence
    - band distributions (quality bands and confidence bands)
    - human review flag rate

    Args:
        results: List of evaluated incident assessment dictionaries or objects.

    Returns:
        dict[str, Any]: Dictionary containing summary metrics.
    """
    total = len(results)
    if total == 0:
        return {
            "total_cases": 0,
            "mean_quality_score": 0.0,
            "mean_confidence": 0.0,
            "band_distribution": {
                "POOR": 0,
                "WEAK": 0,
                "ADEQUATE": 0,
                "STRONG": 0,
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
            },
            "quality_band_distribution": {
                "POOR": 0,
                "WEAK": 0,
                "ADEQUATE": 0,
                "STRONG": 0,
            },
            "confidence_band_distribution": {
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
            },
            "human_review_flag_rate": 0.0,
        }

    quality_scores: list[float] = []
    confidence_values: list[float] = []
    quality_band_counts: dict[str, int] = {
        "POOR": 0,
        "WEAK": 0,
        "ADEQUATE": 0,
        "STRONG": 0,
    }
    confidence_band_counts: dict[str, int] = {
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
    }
    human_review_flags: list[bool] = []

    for res in results:
        qs = _extract_quality_score(res)
        if qs is not None:
            quality_scores.append(qs)

        cv = _extract_confidence_value(res)
        if cv is not None:
            confidence_values.append(cv)

        qb = _extract_quality_band(res)
        if qb in quality_band_counts:
            quality_band_counts[qb] += 1

        cb = _extract_confidence_band(res)
        if cb in confidence_band_counts:
            confidence_band_counts[cb] += 1

        human_review_flags.append(_extract_human_review(res))

    mean_qs = (
        round(statistics.mean(quality_scores), 2)
        if quality_scores
        else 0.0
    )
    mean_conf = (
        round(statistics.mean(confidence_values), 4)
        if confidence_values
        else 0.0
    )
    review_rate = (
        round(sum(1 for f in human_review_flags if f) / total, 4)
    )

    # Combined band distribution mapping both quality and confidence bands
    combined_band_dist: dict[str, int] = {
        **quality_band_counts,
        **confidence_band_counts,
    }

    return {
        "total_cases": total,
        "mean_quality_score": mean_qs,
        "mean_confidence": mean_conf,
        "band_distribution": combined_band_dist,
        "quality_band_distribution": quality_band_counts,
        "confidence_band_distribution": confidence_band_counts,
        "human_review_flag_rate": review_rate,
    }
