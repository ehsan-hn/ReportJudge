"""Automated Evaluation CLI Runner & CI/CD Quality Gate.

Executes incident judgment benchmark cases and invariance property suites against
either offline deterministic mock LLM or live OpenAI models.
Computes quantitative evaluation metrics and enforces mathematical assertions,
serving as an automated quality gate suitable for CI/CD pipelines.

Usage:
    python -m eval.runner --provider fake
    python -m eval.runner --provider openai --model gpt-4o-mini
    python -m eval --provider fake --suite golden
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from app.api.v1.schemas import IncidentInput
from app.config import settings
from app.judgment.llm.factory import get_llm_client
from app.judgment.service import AssessmentResult, AssessmentService
from eval.metrics import (
    compute_floor_enforcement_rate,
    compute_invariance_pass_rate,
    compute_severity_accuracy,
    summarize_evaluation,
)

# Terminal ANSI escape codes for formatted output
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"

BADGE_PASS = f"{COLOR_BOLD}{COLOR_GREEN}[PASS]{COLOR_RESET}"
BADGE_FAIL = f"{COLOR_BOLD}{COLOR_RED}[FAIL]{COLOR_RESET}"
BADGE_WARN = f"{COLOR_BOLD}{COLOR_YELLOW}[WARN]{COLOR_RESET}"
BADGE_INFO = f"{COLOR_BOLD}{COLOR_CYAN}[INFO]{COLOR_RESET}"

DEFAULT_CASES_DIR = Path(__file__).resolve().parent / "cases"


class EvaluationFailure(Exception):
    """Exception raised when an evaluation assertion fails."""


def _normalize_sev(val: Any) -> str:
    """Normalize severity level enum or string."""
    if hasattr(val, "value"):
        return str(val.value).strip().upper()
    return str(val).strip().upper()


def format_table_row(cols: list[tuple[str, int, str]]) -> str:
    """Format a table row with specified column widths and alignments.

    Args:
        cols: List of (text, width, align) tuples where align is '<', '>', or '^'.
    """
    cells = []
    for text, width, align in cols:
        if align == "<":
            cells.append(f"{text:<{width}}")
        elif align == ">":
            cells.append(f"{text:>{width}}")
        else:
            cells.append(f"{text:^{width}}")
    return " | ".join(cells)


async def run_golden_suite(
    service: AssessmentService,
    cases_path: Path,
) -> tuple[bool, list[AssessmentResult], list[dict[str, Any]], list[str]]:
    """Execute Suite 1: Golden Benchmark Suite.

    Args:
        service: AssessmentService configured with target LLM client.
        cases_path: Absolute path to golden.json.

    Returns:
        tuple containing:
            - all_passed (bool)
            - evaluated_results (list[AssessmentResult])
            - raw_cases (list[dict])
            - failure_messages (list[str])
    """
    print(f"\n{COLOR_BOLD}{'=' * 80}{COLOR_RESET}")
    print(f"{COLOR_BOLD}SUITE 1: GOLDEN BENCHMARK SUITE ({cases_path.name}){COLOR_RESET}")
    print(f"{COLOR_BOLD}{'=' * 80}{COLOR_RESET}")

    if not cases_path.exists():
        raise FileNotFoundError(f"Golden cases file not found: {cases_path}")

    with open(cases_path, "r", encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)

    header_cols = [
        ("STATUS", 6, "^"),
        ("CASE ID", 28, "<"),
        ("SEV (EXP/ACT)", 15, "^"),
        ("QUALITY", 10, "^"),
        ("CONFIDENCE", 12, "^"),
        ("REVIEW", 8, "^"),
    ]
    print(format_table_row(header_cols))
    print("-" * 85)

    all_passed = True
    evaluated_results: list[AssessmentResult] = []
    failure_messages: list[str] = []

    for case in cases:
        case_id = case.get("id", "unknown")
        case_name = case.get("name", case_id)
        inp_dict = case.get("input", {})
        expected = case.get("expected", {})

        # Validate input schema boundary
        validated_input = IncidentInput(**inp_dict)

        # Execute assessment through domain service
        res = await service.assess(report=validated_input)
        evaluated_results.append(res)

        case_errors: list[str] = []
        actual_sev = _normalize_sev(res.severity.level)
        exp_sev = _normalize_sev(expected["severity"]) if expected.get("severity") else None

        # Assertion 1: expected_severity
        if exp_sev is not None and actual_sev != exp_sev:
            case_errors.append(f"Severity mismatch: expected {exp_sev}, got {actual_sev}")

        # Assertion 2: expected_floor
        exp_floor = expected.get("expected_floor")
        if exp_floor:
            applied_rules = list(res.severity.applied_rules)
            if exp_floor not in applied_rules:
                case_errors.append(
                    f"Policy floor missing: expected '{exp_floor}' in applied rules {applied_rules}"
                )

        # Assertion 3: expected_min_confidence
        min_conf = expected.get("min_confidence")
        if min_conf is not None and res.confidence.value < min_conf:
            case_errors.append(
                f"Confidence below threshold: expected >= {min_conf}, got {res.confidence.value:.2f}"
            )

        # Assertion 4: expected_max_confidence
        max_conf = expected.get("max_confidence")
        if max_conf is not None and res.confidence.value > max_conf:
            case_errors.append(
                f"Confidence exceeded cap: expected <= {max_conf}, got {res.confidence.value:.2f}"
            )

        # Assertion 5: expected_max_quality
        max_qual = expected.get("max_quality_score")
        if max_qual is not None and res.report_quality.score > max_qual:
            case_errors.append(
                f"Quality score exceeded ceiling: expected <= {max_qual}, got {res.report_quality.score:.1f}"
            )

        # Assertion 5b: min_quality_score (if specified)
        min_qual = expected.get("min_quality_score")
        if min_qual is not None and res.report_quality.score < min_qual:
            case_errors.append(
                f"Quality score below minimum: expected >= {min_qual}, got {res.report_quality.score:.1f}"
            )

        # Assertion 6: must_require_human_review
        req_review_exp = expected.get("must_require_human_review")
        if req_review_exp is not None:
            if req_review_exp is True and not res.requires_human_review:
                case_errors.append("Expected requires_human_review to be True, got False")
            elif req_review_exp is False and res.requires_human_review:
                case_errors.append("Expected requires_human_review to be False, got True")

        # Assertion 7: is_valid_incident_report
        exp_valid = expected.get("is_valid_incident_report")
        if exp_valid is False and res.status != "not_an_incident_report":
            case_errors.append(
                f"Expected status 'not_an_incident_report' for invalid input, got '{res.status}'"
            )

        # Display progress row
        sev_display = f"{exp_sev or '-'}/{actual_sev}"
        qual_display = f"{res.report_quality.score:.1f}"
        conf_display = f"{res.confidence.value:.2f} ({res.confidence.band})"
        review_display = "YES" if res.requires_human_review else "NO"

        if not case_errors:
            status_badge = BADGE_PASS
            row = [
                (status_badge, 6, "^"),
                (case_id[:28], 28, "<"),
                (sev_display, 15, "^"),
                (qual_display, 10, "^"),
                (conf_display, 12, "^"),
                (review_display, 8, "^"),
            ]
            print(format_table_row(row))
        else:
            all_passed = False
            status_badge = BADGE_FAIL
            row = [
                (status_badge, 6, "^"),
                (case_id[:28], 28, "<"),
                (sev_display, 15, "^"),
                (qual_display, 10, "^"),
                (conf_display, 12, "^"),
                (review_display, 8, "^"),
            ]
            print(format_table_row(row))
            for err in case_errors:
                msg = f"  -> [FAIL] Case '{case_id}' ({case_name}): {err}"
                print(f"{COLOR_RED}{msg}{COLOR_RESET}")
                failure_messages.append(msg)

    return all_passed, evaluated_results, cases, failure_messages


async def run_properties_suite(
    service: AssessmentService,
    properties_path: Path,
) -> tuple[bool, list[bool], list[str]]:
    """Execute Suite 2: Invariance & Monotonicity Suite.

    Args:
        service: AssessmentService configured with target LLM client.
        properties_path: Absolute path to properties.json.

    Returns:
        tuple containing:
            - all_passed (bool)
            - invariance_booleans (list[bool])
            - failure_messages (list[str])
    """
    print(f"\n{COLOR_BOLD}{'=' * 80}{COLOR_RESET}")
    print(f"{COLOR_BOLD}SUITE 2: INVARIANCE & MONOTONICITY SUITE ({properties_path.name}){COLOR_RESET}")
    print(f"{COLOR_BOLD}{'=' * 80}{COLOR_RESET}")

    if not properties_path.exists():
        raise FileNotFoundError(f"Properties cases file not found: {properties_path}")

    with open(properties_path, "r", encoding="utf-8") as f:
        properties: list[dict[str, Any]] = json.load(f)

    all_passed = True
    invariance_results: list[bool] = []
    failure_messages: list[str] = []

    for prop in properties:
        prop_id = prop.get("property_id", "unknown")
        prop_name = prop.get("name", prop_id)
        cases = prop.get("cases", {})
        invariant_type = prop.get("invariant_type", "")

        print(f"\n{COLOR_CYAN}Property: [{prop_id}] {prop_name}{COLOR_RESET}")
        print(f"  Description: {prop.get('description', '')}")

        prop_passed = True
        err_details: list[str] = []

        # 1. Evidence Monotonicity
        if invariant_type == "monotonicity_evidence" or prop_id == "monotonicity_evidence":
            weak_inp = IncidentInput(**cases["weak"])
            strong_inp = IncidentInput(**cases["strong"])

            weak_res = await service.assess(report=weak_inp)
            strong_res = await service.assess(report=strong_inp)

            diff = strong_res.confidence.value - weak_res.confidence.value
            print(
                f"  Evidence Monotonicity: weak.conf={weak_res.confidence.value:.4f} "
                f"-> strong.conf={strong_res.confidence.value:.4f} (delta: {diff:+.4f})"
            )

            if strong_res.confidence.value < weak_res.confidence.value:
                prop_passed = False
                err_details.append(
                    f"Evidence monotonicity violated: strong confidence ({strong_res.confidence.value:.4f}) "
                    f"< weak confidence ({weak_res.confidence.value:.4f})"
                )

        # 2. Field Coverage Monotonicity
        elif invariant_type == "monotonicity_field_coverage" or prop_id == "monotonicity_field_coverage":
            minimal_inp = IncidentInput(**cases["minimal"])
            complete_inp = IncidentInput(**cases["complete"])

            minimal_res = await service.assess(report=minimal_inp)
            complete_res = await service.assess(report=complete_inp)

            diff = complete_res.confidence.value - minimal_res.confidence.value
            print(
                f"  Coverage Monotonicity: minimal.conf={minimal_res.confidence.value:.4f} "
                f"-> complete.conf={complete_res.confidence.value:.4f} (delta: {diff:+.4f})"
            )

            if complete_res.confidence.value < minimal_res.confidence.value:
                prop_passed = False
                err_details.append(
                    f"Field coverage monotonicity violated: complete confidence ({complete_res.confidence.value:.4f}) "
                    f"< minimal confidence ({minimal_res.confidence.value:.4f})"
                )

        # 3. Policy Floor Invariance (Breach)
        elif invariant_type == "policy_floor_invariance" or prop_id == "breach_policy_floor_invariance":
            exp_sev = prop.get("expected_severity", "SEV1")
            exp_floor = prop.get("expected_floor", "FLOOR_DATA_LOSS_OR_BREACH_SEV1")

            for variant_name, case_data in cases.items():
                var_inp = IncidentInput(**case_data)
                var_res = await service.assess(report=var_inp)
                actual_sev = _normalize_sev(var_res.severity.level)
                applied_floors = list(var_res.severity.applied_rules)

                print(
                    f"  Variant '{variant_name}': severity={actual_sev} "
                    f"floors={applied_floors}"
                )

                if actual_sev != exp_sev:
                    prop_passed = False
                    err_details.append(
                        f"Variant '{variant_name}' severity mismatch: expected {exp_sev}, got {actual_sev}"
                    )
                if exp_floor not in applied_floors:
                    prop_passed = False
                    err_details.append(
                        f"Variant '{variant_name}' missing expected floor '{exp_floor}' in {applied_floors}"
                    )

        else:
            print(f"  {BADGE_WARN} Unknown property invariant type: '{invariant_type}'. Skipped.")

        invariance_results.append(prop_passed)
        if prop_passed:
            print(f"  Result: {BADGE_PASS} Assertion satisfied.")
        else:
            all_passed = False
            print(f"  Result: {BADGE_FAIL} Invariant check failed:")
            for err in err_details:
                msg = f"    - {err}"
                print(f"{COLOR_RED}{msg}{COLOR_RESET}")
                failure_messages.append(f"Property '{prop_id}': {err}")

    return all_passed, invariance_results, failure_messages


def print_summary_report(
    provider: str,
    model: str,
    suite: str,
    golden_results: list[AssessmentResult],
    golden_cases: list[dict[str, Any]],
    invariance_results: list[bool],
    failures: list[str],
) -> None:
    """Print comprehensive summary report with quantitative benchmark metrics."""
    print(f"\n{COLOR_BOLD}{'=' * 80}{COLOR_RESET}")
    print(f"{COLOR_BOLD}EVALUATION BENCHMARK SUMMARY REPORT{COLOR_RESET}")
    print(f"{COLOR_BOLD}{'=' * 80}{COLOR_RESET}")
    print(f"Provider:            {COLOR_CYAN}{provider}{COLOR_RESET}")
    print(f"Model:               {COLOR_CYAN}{model}{COLOR_RESET}")
    print(f"Evaluated Suite:     {COLOR_CYAN}{suite}{COLOR_RESET}")
    print("-" * 80)

    # 1. Severity Accuracy
    if golden_results and golden_cases:
        preds = []
        tgts = []
        for res, case in zip(golden_results, golden_cases):
            exp_sev = case.get("expected", {}).get("severity")
            if exp_sev:
                preds.append(_normalize_sev(res.severity.level))
                tgts.append(_normalize_sev(exp_sev))

        if tgts:
            accuracy = compute_severity_accuracy(preds, tgts)
            print(f"Severity Accuracy:          {accuracy * 100:.1f}% ({int(round(accuracy * len(tgts)))}/{len(tgts)})")

        # 2. Policy Floor Enforcement Rate
        expected_floors = [case.get("expected", {}).get("expected_floor") for case in golden_cases]
        floor_rate = compute_floor_enforcement_rate(
            [{"severity": {"applied_rules": r.severity.applied_rules}} for r in golden_results],
            expected_floors,
        )
        total_expected_floors = sum(1 for f in expected_floors if f is not None)
        print(f"Policy Floor Enforcement:   {floor_rate * 100:.1f}% (Required floors: {total_expected_floors})")

        # 3. Distribution Metrics
        summary = summarize_evaluation(
            [
                {
                    "report_quality": {"score": r.report_quality.score, "band": r.report_quality.band},
                    "confidence": {"value": r.confidence.value, "band": r.confidence.band},
                    "requires_human_review": r.requires_human_review,
                }
                for r in golden_results
            ]
        )

        print(f"Total Evaluated Cases:      {summary['total_cases']}")
        print(f"Mean Quality Score:         {summary['mean_quality_score']:.1f} / 100.0")
        print(f"Mean Calibrated Confidence: {summary['mean_confidence']:.4f}")
        print(f"Human Review Flag Rate:     {summary['human_review_flag_rate'] * 100:.1f}%")
        print(f"Quality Band Breakdown:     {summary['quality_band_distribution']}")
        print(f"Confidence Band Breakdown:  {summary['confidence_band_distribution']}")

    # 4. Invariance Pass Rate
    if invariance_results:
        inv_pass_rate = compute_invariance_pass_rate(invariance_results)
        passed_count = sum(1 for r in invariance_results if r)
        print(f"Invariance Property Pass:   {inv_pass_rate * 100:.1f}% ({passed_count}/{len(invariance_results)})")

    print("-" * 80)
    if not failures:
        print(
            f"{COLOR_BOLD}{COLOR_GREEN}[ALL BENCHMARKS & INVARIANCE PROPERTIES PASSED]{COLOR_RESET}"
        )
    else:
        print(
            f"{COLOR_BOLD}{COLOR_RED}[EVALUATION BENCHMARK FAILED - {len(failures)} ASSERTION FAILURE(S)]{COLOR_RESET}"
        )
        for idx, failure in enumerate(failures, 1):
            print(f"  {idx}. {failure}")
    print(f"{COLOR_BOLD}{'=' * 80}{COLOR_RESET}\n")


async def run_evaluation(
    provider: str = settings.llm_provider,
    model: str = settings.openai_model,
    suite: str = "all",
    api_key: str | None = None,
    cases_dir: Path | None = None,
) -> int:
    """Run evaluation suites and return CI exit code (0 on success, 1 on failure).

    Args:
        provider: LLM provider ("fake" or "openai").
        model: Model identifier.
        suite: Suite selection ("all", "golden", or "properties").
        api_key: Optional OpenAI API key override.
        cases_dir: Optional custom path to benchmark cases directory.

    Returns:
        int: 0 if all assertions and invariants pass, 1 if any failure occurs.
    """
    resolved_cases_dir = cases_dir if cases_dir is not None else DEFAULT_CASES_DIR
    golden_path = resolved_cases_dir / "golden.json"
    properties_path = resolved_cases_dir / "properties.json"

    # Initialize LLM client via provider factory
    effective_api_key = api_key if api_key is not None else settings.openai_api_key
    try:
        client = get_llm_client(
            provider=provider,
            api_key=effective_api_key,
            model=model,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    except Exception as exc:
        print(f"{COLOR_RED}Error initializing LLM client ({provider}): {exc}{COLOR_RESET}", file=sys.stderr)
        return 1

    service = AssessmentService(llm_client=client)

    all_golden_passed = True
    all_props_passed = True
    golden_results: list[AssessmentResult] = []
    golden_cases: list[dict[str, Any]] = []
    invariance_results: list[bool] = []
    all_failures: list[str] = []

    # Run Suite 1: Golden Benchmark Suite
    if suite in ("all", "golden"):
        try:
            (
                all_golden_passed,
                golden_results,
                golden_cases,
                golden_failures,
            ) = await run_golden_suite(service, golden_path)
            all_failures.extend(golden_failures)
        except Exception as exc:
            all_golden_passed = False
            msg = f"Golden Suite Execution Error: {exc}"
            print(f"{COLOR_RED}{msg}{COLOR_RESET}", file=sys.stderr)
            all_failures.append(msg)

    # Run Suite 2: Invariance & Monotonicity Suite
    if suite in ("all", "properties"):
        try:
            (
                all_props_passed,
                invariance_results,
                props_failures,
            ) = await run_properties_suite(service, properties_path)
            all_failures.extend(props_failures)
        except Exception as exc:
            all_props_passed = False
            msg = f"Properties Suite Execution Error: {exc}"
            print(f"{COLOR_RED}{msg}{COLOR_RESET}", file=sys.stderr)
            all_failures.append(msg)

    # Print Aggregate Summary
    print_summary_report(
        provider=provider,
        model=model,
        suite=suite,
        golden_results=golden_results,
        golden_cases=golden_cases,
        invariance_results=invariance_results,
        failures=all_failures,
    )

    is_success = all_golden_passed and all_props_passed and len(all_failures) == 0
    return 0 if is_success else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for eval runner."""
    parser = argparse.ArgumentParser(
        prog="python -m eval.runner",
        description="Automated evaluation benchmark runner and CI/CD quality gate for incident scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["fake", "openai"],
        default=settings.llm_provider,
        help="LLM provider backend to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.openai_model,
        help="LLM model identifier to benchmark.",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "golden", "properties"],
        default="all",
        help="Evaluation suite to execute.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (optional; defaults to environment / settings).",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=None,
        help="Custom directory containing golden.json and properties.json.",
    )
    return parser


async def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point returning status code."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    exit_code = await run_evaluation(
        provider=args.provider,
        model=args.model,
        suite=args.suite,
        api_key=args.api_key,
        cases_dir=args.cases_dir,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
