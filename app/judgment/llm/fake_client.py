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
            or "disregard all" in prompt_lower
            or "system prompt override" in prompt_lower
            or "ignore rubric" in prompt_lower
            or "override" in prompt_lower
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
            for k in ["datadog", "5xx", "guardduty", "latency", "prometheus", "alert", "504 gateway"]
        )
        has_screenshot_or_ticket = any(
            k in prompt_lower
            for k in ["screenshot", "ticket", "ui-", "viewport", "safari"]
        )
        is_breach = any(
            k in prompt_lower
            for k in ["credential", "password", "exfiltration", "s3 bucket", "public s3", "compromised"]
        ) or ("breach" in prompt_lower and "memory leak" not in prompt_lower) or ("leak" in prompt_lower and "memory leak" not in prompt_lower)

        is_total_outage = any(
            k in prompt_lower
            for k in ["total outage", "complete outage", "100% of customer", "all user logins", "all users are completely unable", "failing globally", "auth-service"]
        )
        is_checkout = any(
            k in prompt_lower
            for k in ["checkout", "payment", "billing", "cart"]
        )
        is_partial = any(
            k in prompt_lower
            for k in ["partial degradation", "workaround available", "workaround", "slow image"]
        )
        is_cosmetic = any(
            k in prompt_lower
            for k in ["cosmetic", "footer alignment", "alignment glitch", "overlap slightly", "8 pixels"]
        )

        # Synthesize Quality Dimensions
        if has_metrics:
            evidence_score = 4
            evidence_confidence = 0.90
            evidence_justification = "Concrete monitoring telemetry, alerts, or error metrics cited in the incident report."
        elif has_screenshot_or_ticket:
            evidence_score = 3
            evidence_confidence = 0.80
            evidence_justification = "Visual screenshot, ticket link, or client UI artifact provided as evidence."
        else:
            evidence_score = 1
            evidence_confidence = 0.70
            evidence_justification = "Report lacks empirical monitoring metrics, error logs, or telemetry data."

        if is_breach:
            impact_score = 4
            impact_confidence = 0.90
            impact_justification = "Direct data breach or compromised credential exposure documented."
        elif is_total_outage:
            impact_score = 4
            impact_confidence = 0.90
            impact_justification = "Global service outage affecting entire user base with complete loss of functionality."
        elif is_checkout:
            impact_score = 3
            impact_confidence = 0.85
            impact_justification = "Critical commercial impact affecting core revenue and checkout funnels."
        elif is_partial or is_cosmetic:
            impact_score = 3
            impact_confidence = 0.80
            impact_justification = "Scoped impact clearly articulated with non-critical business disruption."
        else:
            impact_score = 1
            impact_confidence = 0.70
            impact_justification = "User or business impact is vague, localized, or unquantified."

        if has_metrics or has_screenshot_or_ticket or is_partial or is_cosmetic:
            clarity_score = 4
            clarity_confidence = 0.85
            clarity_justification = "Clear technical terminology and coherent operational narrative."
        else:
            clarity_score = 2
            clarity_confidence = 0.70
            clarity_justification = "Vague narrative with limited technical precision."

        if has_metrics:
            repro_score = 3
            repro_confidence = 0.80
            repro_justification = "Failure conditions or trigger events identifiable from telemetry."
        elif has_screenshot_or_ticket or is_partial:
            repro_score = 2
            repro_confidence = 0.75
            repro_justification = "Specific client environment or file conditions described."
        else:
            repro_score = 1
            repro_confidence = 0.65
            repro_justification = "No specific steps or trigger payload identified to reproduce the issue."

        if has_metrics or has_screenshot_or_ticket or is_partial:
            action_score = 3
            action_confidence = 0.80
            action_justification = "Investigation or initial remediation context documented."
        else:
            action_score = 1
            action_confidence = 0.65
            action_justification = "Zero mitigation actions or diagnostic troubleshooting steps documented."

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
        if is_breach:
            impact_scope = 4
            business_criticality = 4
            time_sensitivity = 2
            severity_confidence = 0.90
            severity_reasoning = "Security breach or credential exposure posing severe data integrity and compliance risk."
        elif is_total_outage:
            impact_scope = 4
            business_criticality = 2
            time_sensitivity = 1
            severity_confidence = 0.90
            severity_reasoning = "Total outage of primary service core flow affecting all users globally."
        elif is_checkout:
            impact_scope = 3
            business_criticality = 3
            time_sensitivity = 2
            severity_confidence = 0.90
            severity_reasoning = "Active degradation in commercial checkout flow directly impairing transactions and revenue."
        elif is_partial:
            impact_scope = 2
            business_criticality = 1
            time_sensitivity = 1
            severity_confidence = 0.85
            severity_reasoning = "Partial service degradation affecting subset of users with active workaround."
        elif is_cosmetic:
            impact_scope = 1
            business_criticality = 0
            time_sensitivity = 0
            severity_confidence = 0.80
            severity_reasoning = "Minor cosmetic visual rendering glitch without functional or transaction loss."
        elif has_metrics:
            impact_scope = 1
            business_criticality = 1
            time_sensitivity = 0
            severity_confidence = 0.90
            severity_reasoning = "Operational anomaly with telemetry signals; scoped failure without confirmed revenue halt."
        else:
            impact_scope = 1
            business_criticality = 1
            time_sensitivity = 0
            severity_confidence = 0.70
            severity_reasoning = "Anecdotal or localized issue with unverified impact; default baseline single-user scope."

        severity_signals = SeveritySignals(
            reasoning=severity_reasoning,
            impact_scope=impact_scope,
            business_criticality=business_criticality,
            time_sensitivity=time_sensitivity,
            confidence=severity_confidence,
        )

        # Missing Information
        if has_metrics or has_screenshot_or_ticket or is_partial:
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
        elif is_total_outage:
            summary = "Total outage of core authentication and service flows."
        elif is_checkout:
            summary = "Commercial incident degrading checkout and payment flows."
        elif is_partial:
            summary = "Partial degradation with workaround available."
        elif is_cosmetic:
            summary = "Cosmetic alignment issue on mobile device."
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
