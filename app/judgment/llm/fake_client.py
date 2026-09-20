"""Offline Deterministic Mock LLM Client.

Provides dual-mode execution for testing and local development:
1. Exact FIFO replay of scripted LLMAssessment responses for deterministic unit tests.
2. Rule-based heuristic fallback that inspects prompt keywords to synthesize calibrated,
   Pydantic-valid assessments without network or external LLM dependencies.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
"""

from typing import Any

from ..contracts import DimensionSignal, LLMAssessment, SeveritySignals
from ..rubric import MissingInfoCode, QualityDimension
from .base import LLMClient


class FakeLLMClient(LLMClient):
    """Offline mock LLM client with scripted FIFO queue and heuristic fallback."""

    def __init__(
        self,
        scripted_responses: list[LLMAssessment] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FakeLLMClient with optional scripted responses.

        Args:
            scripted_responses: Optional list of pre-canned LLMAssessment responses
                that will be popped and returned FIFO.
            **kwargs: Ignored keyword arguments for compatibility with provider factory.
        """
        self._script: list[LLMAssessment] = (
            list(scripted_responses) if scripted_responses is not None else []
        )

    def queue_response(self, response: LLMAssessment) -> None:
        """Enqueue a scripted assessment response to the FIFO queue."""
        self._script.append(response)

    def queue_responses(self, responses: list[LLMAssessment]) -> None:
        """Enqueue multiple scripted assessment responses to the FIFO queue."""
        self._script.extend(responses)

    @property
    def remaining_scripted_responses(self) -> int:
        """Return the count of remaining scripted responses in the queue."""
        return len(self._script)

    async def complete_assessment(
        self, system_prompt: str, user_prompt: str
    ) -> LLMAssessment:
        """Completes semantic assessment of the incident report.

        Mode 1 (Scripted Queue): Pops and returns the next response from FIFO queue.
        Mode 2 (Heuristic Fallback): Inspects user_prompt keywords to synthesize a
            deterministic, schema-compliant LLMAssessment.

        Args:
            system_prompt: The system evaluation prompt.
            user_prompt: The sanitized user report prompt.

        Returns:
            LLMAssessment: Validated assessment response.
        """
        # Mode 1: FIFO Scripted Replay
        if self._script:
            return self._script.pop(0)

        # Mode 2: Heuristic Fallback
        prompt_lower = user_prompt.lower()
        has_report_tags = "<report>" in prompt_lower and "</report>" in prompt_lower
        is_short_without_tags = len(user_prompt.strip()) < 30 and not has_report_tags

        # Adversarial / Non-incident checks
        is_adversarial_or_invalid = (
            "ignore all" in prompt_lower
            or "hello" in prompt_lower
            or is_short_without_tags
        )

        if is_adversarial_or_invalid:
            dimensions = {
                dim: DimensionSignal(
                    justification="Input was identified as invalid, conversational, or an adversarial injection attempt.",
                    evidence_quote=None,
                    score=0,
                    confidence=0.0,
                )
                for dim in QualityDimension
            }
            severity_signals = SeveritySignals(
                reasoning="Input does not constitute a valid technical incident report; severity evaluation inapplicable.",
                impact_scope=0,
                business_criticality=0,
                time_sensitivity=0,
                confidence=0.0,
            )
            return LLMAssessment(
                is_valid_incident_report=False,
                dimensions=dimensions,
                severity_signals=severity_signals,
                missing_information=[],
                missing_information_notes="Report failed validity checks.",
                summary="The submitted input is not a valid technical incident report (conversational, adversarial, or lacks incident context).",
            )

        # Heuristic Signal Detection
        has_metrics = any(
            k in prompt_lower
            for k in ["datadog", "5xx", "guardduty", "latency", "prometheus", "alert"]
        )
        is_checkout = any(
            k in prompt_lower
            for k in ["checkout", "payment", "billing", "cart"]
        )
        is_breach = any(
            k in prompt_lower
            for k in ["breach", "credential", "password", "leak", "compromised", "s3 bucket"]
        )

        # Synthesize Quality Dimensions
        evidence_score = 4 if has_metrics else 1
        evidence_confidence = 0.90 if has_metrics else 0.70
        evidence_justification = (
            "Concrete monitoring telemetry, alerts, or error metrics cited in the incident report."
            if has_metrics
            else "Report lacks empirical monitoring metrics, error logs, or telemetry data."
        )

        impact_score = 4 if is_breach else (3 if is_checkout else 1)
        impact_confidence = 0.90 if is_breach else (0.85 if is_checkout else 0.70)
        impact_justification = (
            "Direct data breach or compromised credential exposure documented."
            if is_breach
            else (
                "Critical commercial impact affecting core revenue and checkout funnels."
                if is_checkout
                else "User or business impact is vague, localized, or unquantified."
            )
        )

        clarity_score = 4 if has_metrics else 2
        clarity_confidence = 0.85 if has_metrics else 0.70
        clarity_justification = (
            "Clear technical terminology and coherent operational narrative."
            if has_metrics
            else "Vague narrative with limited technical precision."
        )

        repro_score = 3 if has_metrics else 1
        repro_confidence = 0.80 if has_metrics else 0.65
        repro_justification = (
            "Failure conditions or trigger events identifiable from telemetry."
            if has_metrics
            else "No specific steps or trigger payload identified to reproduce the issue."
        )

        action_score = 3 if has_metrics else 1
        action_confidence = 0.80 if has_metrics else 0.65
        action_justification = (
            "Investigation or initial remediation context documented."
            if has_metrics
            else "Zero mitigation actions or diagnostic troubleshooting steps documented."
        )

        dimensions = {
            QualityDimension.CLARITY: DimensionSignal(
                justification=clarity_justification,
                evidence_quote=None,
                score=clarity_score,
                confidence=clarity_confidence,
            ),
            QualityDimension.EVIDENCE_STRENGTH: DimensionSignal(
                justification=evidence_justification,
                evidence_quote=None,
                score=evidence_score,
                confidence=evidence_confidence,
            ),
            QualityDimension.IMPACT_ARTICULATION: DimensionSignal(
                justification=impact_justification,
                evidence_quote=None,
                score=impact_score,
                confidence=impact_confidence,
            ),
            QualityDimension.REPRODUCIBILITY: DimensionSignal(
                justification=repro_justification,
                evidence_quote=None,
                score=repro_score,
                confidence=repro_confidence,
            ),
            QualityDimension.ACTION_CONTEXT: DimensionSignal(
                justification=action_justification,
                evidence_quote=None,
                score=action_score,
                confidence=action_confidence,
            ),
        }

        # Synthesize Severity Signals
        impact_scope = 4 if is_breach else (3 if is_checkout else 1)
        business_criticality = 4 if is_breach else (3 if is_checkout else 1)
        time_sensitivity = 2 if (is_checkout or is_breach) else 0
        severity_confidence = 0.90 if has_metrics else 0.70

        if is_breach:
            severity_reasoning = (
                "Security breach or credential exposure posing severe data integrity and compliance risk."
            )
        elif is_checkout:
            severity_reasoning = (
                "Active degradation in commercial checkout flow directly impairing transactions and revenue."
            )
        elif has_metrics:
            severity_reasoning = (
                "Operational anomaly with telemetry signals; scoped failure without confirmed revenue halt."
            )
        else:
            severity_reasoning = (
                "Anecdotal or localized issue with unverified impact; default baseline single-user scope."
            )

        severity_signals = SeveritySignals(
            reasoning=severity_reasoning,
            impact_scope=impact_scope,
            business_criticality=business_criticality,
            time_sensitivity=time_sensitivity,
            confidence=severity_confidence,
        )

        # Missing Information
        if has_metrics:
            missing_info: list[MissingInfoCode] = []
            missing_notes = ""
        else:
            missing_info = [
                MissingInfoCode.NO_MONITORING_DATA,
                MissingInfoCode.IMPACT_SCOPE_UNQUANTIFIED,
            ]
            missing_notes = (
                "Report lacks telemetry/monitoring metrics and quantified user/business impact."
            )

        # Summary
        if is_breach:
            summary = "Critical security incident with confirmed or suspected data compromise."
        elif is_checkout:
            summary = "Commercial incident degrading checkout and payment flows."
        elif has_metrics:
            summary = "Operational incident supported by concrete telemetry and monitoring alerts."
        else:
            summary = "Vague incident report lacking monitoring data and quantified impact."

        return LLMAssessment(
            is_valid_incident_report=True,
            dimensions=dimensions,
            severity_signals=severity_signals,
            missing_information=missing_info,
            missing_information_notes=missing_notes,
            summary=summary,
        )
