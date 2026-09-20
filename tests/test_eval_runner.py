"""Unit, Integration, and CLI Tests for Automated Evaluation Runner.

Tests:
1. Programmatic execution of eval runner with fake provider across all suites.
2. Direct execution and validation of golden benchmark suite.
3. Direct execution and validation of invariance property suite.
4. CI quality gate failure detection and exit code 1 return on assertion failures.
5. Error handling for missing files and missing API keys.
6. Subprocess execution via 'python -m eval.runner' and 'python -m eval'.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest

from app.judgment.llm.fake_client import FakeLLMClient
from app.judgment.service import AssessmentService
from eval.runner import (
    DEFAULT_CASES_DIR,
    build_parser,
    main,
    run_evaluation,
    run_golden_suite,
    run_properties_suite,
)

CASES_DIR = Path(__file__).resolve().parent.parent / "eval" / "cases"
GOLDEN_PATH = CASES_DIR / "golden.json"
PROPERTIES_PATH = CASES_DIR / "properties.json"


# ==============================================================================
# 1. Programmatic Runner Tests (main entry point)
# ==============================================================================


@pytest.mark.asyncio
async def test_eval_runner_main_fake_all_suites():
    """Executing runner with provider='fake' across all suites returns exit code 0."""
    exit_code = await main(["--provider", "fake", "--suite", "all"])
    assert exit_code == 0


@pytest.mark.asyncio
async def test_eval_runner_main_fake_golden_only():
    """Executing runner on golden suite only returns exit code 0."""
    exit_code = await main(["--provider", "fake", "--suite", "golden"])
    assert exit_code == 0


@pytest.mark.asyncio
async def test_eval_runner_main_fake_properties_only():
    """Executing runner on properties suite only returns exit code 0."""
    exit_code = await main(["--provider", "fake", "--suite", "properties"])
    assert exit_code == 0


@pytest.mark.asyncio
async def test_eval_runner_custom_cases_dir(tmp_path: Path):
    """Executing runner with custom cases-dir operates correctly."""
    # Copy valid cases to tmp directory
    golden_tmp = tmp_path / "golden.json"
    props_tmp = tmp_path / "properties.json"
    golden_tmp.write_text(GOLDEN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    props_tmp.write_text(PROPERTIES_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    exit_code = await main(
        ["--provider", "fake", "--cases-dir", str(tmp_path), "--suite", "all"]
    )
    assert exit_code == 0


# ==============================================================================
# 2. Direct Suite Execution Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_run_golden_suite_direct():
    """Direct execution of run_golden_suite returns all_passed=True and 6 results."""
    service = AssessmentService(llm_client=FakeLLMClient())
    all_passed, results, cases, failures = await run_golden_suite(service, GOLDEN_PATH)

    assert all_passed is True
    assert len(results) == len(cases) == 6
    assert failures == []

    # Check key expectations were validated
    res_map = {cases[i]["id"]: results[i] for i in range(len(cases))}

    # SEV1 credential leak
    sev1_res = res_map["sev1_credential_leak"]
    assert str(sev1_res.severity.level.value) == "SEV1"
    assert "FLOOR_DATA_LOSS_OR_BREACH_SEV1" in sev1_res.severity.applied_rules
    assert sev1_res.requires_human_review is True

    # SEV2 total outage
    sev2_res = res_map["sev2_total_outage_core_flow"]
    assert str(sev2_res.severity.level.value) == "SEV2"
    assert "FLOOR_TOTAL_OUTAGE_SEV2" in sev2_res.severity.applied_rules
    assert sev2_res.requires_human_review is True

    # Adversarial injection
    adv_res = res_map["adversarial_injection"]
    assert adv_res.status == "not_an_incident_report"
    assert adv_res.report_quality.score == 0.0
    assert adv_res.requires_human_review is True


@pytest.mark.asyncio
async def test_run_properties_suite_direct():
    """Direct execution of run_properties_suite validates all invariant properties."""
    service = AssessmentService(llm_client=FakeLLMClient())
    all_passed, inv_results, failures = await run_properties_suite(service, PROPERTIES_PATH)

    assert all_passed is True
    assert len(inv_results) == 3
    assert all(inv_results)
    assert failures == []


# ==============================================================================
# 3. Quality Gate Failure and Assertion Enforcement Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_golden_suite_assertion_failure_detection(tmp_path: Path):
    """When a benchmark case expectation fails, the runner reports failure and exit code 1."""
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Artificially modify the first case to require an impossible expectation
    cases[0]["expected"]["severity"] = "SEV4"  # Expected was SEV1
    cases[0]["expected"]["min_confidence"] = 0.99  # Actual is ~0.74

    tampered_golden = tmp_path / "golden.json"
    tampered_golden.write_text(json.dumps(cases), encoding="utf-8")

    service = AssessmentService(llm_client=FakeLLMClient())
    all_passed, results, _, failures = await run_golden_suite(service, tampered_golden)

    assert all_passed is False
    assert len(failures) >= 2
    assert any("Severity mismatch" in f for f in failures)
    assert any("Confidence below threshold" in f for f in failures)


@pytest.mark.asyncio
async def test_properties_suite_assertion_failure_detection(tmp_path: Path):
    """When an invariance property assertion fails, the runner reports failure."""
    with open(PROPERTIES_PATH, "r", encoding="utf-8") as f:
        props = json.load(f)

    # Invert the evidence monotonicity cases: swap weak and strong
    for p in props:
        if p.get("property_id") == "monotonicity_evidence":
            original_cases = p["cases"]
            p["cases"] = {
                "weak": original_cases["strong"],
                "strong": original_cases["weak"],
            }

    tampered_props = tmp_path / "properties.json"
    tampered_props.write_text(json.dumps(props), encoding="utf-8")

    service = AssessmentService(llm_client=FakeLLMClient())
    all_passed, inv_results, failures = await run_properties_suite(service, tampered_props)

    assert all_passed is False
    assert False in inv_results
    assert len(failures) >= 1
    assert any("monotonicity violated" in f.lower() for f in failures)


@pytest.mark.asyncio
async def test_run_evaluation_exit_code_on_failure(tmp_path: Path):
    """run_evaluation returns 1 if any suite fails."""
    # Write a failing golden case
    failing_cases = [
        {
            "id": "always_fails",
            "name": "Failing Case",
            "input": {
                "title": "Sample test incident",
                "description": "Sample test description of failure",
                "impact": None,
                "evidence": None,
                "actions_taken": None,
            },
            "expected": {
                "severity": "SEV1",  # Will evaluate to SEV4
                "expected_floor": "FLOOR_DATA_LOSS_OR_BREACH_SEV1",
                "min_confidence": 0.95,
            },
        }
    ]
    tampered_golden = tmp_path / "golden.json"
    tampered_golden.write_text(json.dumps(failing_cases), encoding="utf-8")

    exit_code = await run_evaluation(
        provider="fake",
        suite="golden",
        cases_dir=tmp_path,
    )
    assert exit_code == 1


# ==============================================================================
# 4. Error Handling and Parameter Validation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_missing_cases_file_handled_gracefully(tmp_path: Path):
    """Missing cases files result in failure exit code 1 without unhandled crash."""
    empty_dir = tmp_path / "empty_cases"
    empty_dir.mkdir()

    exit_code = await run_evaluation(
        provider="fake",
        suite="all",
        cases_dir=empty_dir,
    )
    assert exit_code == 1


@pytest.mark.asyncio
async def test_openai_provider_without_api_key_fails_cleanly():
    """Evaluating with openai provider when no API key is available returns 1."""
    with patch("eval.runner.settings.openai_api_key", None):
        exit_code = await run_evaluation(
            provider="openai",
            api_key=None,
            suite="golden",
        )
        assert exit_code == 1


def test_cli_parser_defaults():
    """Parser assigns expected defaults from application configuration."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.provider in ("fake", "openai")
    assert args.model == "gpt-4o-mini"
    assert args.suite == "all"
    assert args.cases_dir is None


def test_cli_parser_custom_arguments():
    """Parser correctly parses custom CLI arguments."""
    parser = build_parser()
    args = parser.parse_args([
        "--provider", "openai",
        "--model", "gpt-4o",
        "--suite", "properties",
        "--api-key", "test-key-123",
        "--cases-dir", "/tmp/custom_cases",
    ])

    assert args.provider == "openai"
    assert args.model == "gpt-4o"
    assert args.suite == "properties"
    assert args.api_key == "test-key-123"
    assert args.cases_dir == Path("/tmp/custom_cases")


# ==============================================================================
# 5. End-to-End Subprocess CLI Execution Tests
# ==============================================================================


def test_subprocess_eval_runner_cli():
    """Executing 'python -m eval.runner --provider fake' via subprocess exits with 0."""
    result = subprocess.run(
        [sys.executable, "-m", "eval.runner", "--provider", "fake"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "[ALL BENCHMARKS & INVARIANCE PROPERTIES PASSED]" in result.stdout
    assert "Severity Accuracy:          100.0%" in result.stdout
    assert "Invariance Property Pass:   100.0%" in result.stdout


def test_subprocess_eval_module_cli():
    """Executing 'python -m eval --provider fake' via subprocess exits with 0."""
    result = subprocess.run(
        [sys.executable, "-m", "eval", "--provider", "fake"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "[ALL BENCHMARKS & INVARIANCE PROPERTIES PASSED]" in result.stdout
