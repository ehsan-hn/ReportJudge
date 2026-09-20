"""Central Domain Assessment Orchestrator.

Pure domain orchestrator coordinating semantic LLM perception, deterministic quality
scoring, policy floor severity calculation, and mathematical confidence bounding.

Strict Architectural Rule: ZERO framework (FastAPI/Starlette) or HTTP dependencies.
Can be executed in CLI, async background workers, AWS Lambda, or unit tests without
mocking HTTP requests.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import core_schema

from .confidence import (
    BAND_LOW,
    LIMITING_REASON_WEAK_EVIDENCE,
    MIN_CONFIDENCE,
    compute_confidence,
)
from .contracts import LLMAssessment
from .llm.base import LLMClient
from .llm.fake_client import FakeLLMClient
from .prompts import format_assessment_prompt
from .rubric import (
    QUALITY_WEIGHTS,
    MissingInfoCode,
    QualityDimension,
    SeverityLevel,
)
from .scoring import compute_quality_score, compute_severity


class AttrDict(dict):
    """Dictionary subclass supporting both item indexing (d['k']) and attribute access (d.k)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(lambda v: AttrDict(v))


class AssessmentResult(BaseModel):
    """Domain model packaging the complete incident assessment outcome.

    Supports both attribute access (result.status) and dictionary indexing (result['status']).
    """

    model_config = ConfigDict(extra="ignore")

    status: Literal["assessed", "not_an_incident_report"]
    report_quality: AttrDict
    severity: AttrDict
    confidence: AttrDict
    missing_information: list[MissingInfoCode] = Field(default_factory=list)
    missing_information_notes: str = ""
    summary_explanation: str = ""
    requires_human_review: bool

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)


class AssessmentService:
    """Central domain orchestrator for incident assessment and scoring."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialize AssessmentService with an LLM client.

        Args:
            llm_client: Implementation of LLMClient. If None, instantiates FakeLLMClient.
        """
        self.llm_client: LLMClient = llm_client if llm_client is not None else FakeLLMClient()

    async def assess(
        self,
        title: str | Any = None,
        description: str | None = None,
        impact: str | None = None,
        evidence: str | None = None,
        actions_taken: str | None = None,
        *,
        report: Any = None,
        **kwargs: Any,
    ) -> AssessmentResult:
        """Assess an incident report through the multi-stage evaluation pipeline.

        Accepts either individual arguments, keyword arguments, or any duck-typed
        object / dict containing incident report fields.

        Args:
            title: Incident title or duck-typed report object.
            description: Incident narrative / symptom description.
            impact: Optional user / business impact description.
            evidence: Optional logs / metrics / alert telemetry.
            actions_taken: Optional troubleshooting / mitigation steps taken.
            report: Optional duck-typed object passed as keyword argument.
            **kwargs: Extra fields (e.g. actions alias).

        Returns:
            AssessmentResult: Complete structured evaluation with deterministic scores.
        """
        # 0. Extract raw fields supporting duck-typed objects, dicts, or individual args
        target_obj = report if report is not None else title

        if isinstance(target_obj, dict):
            raw_title = target_obj.get("title", "")
            raw_desc = target_obj.get("description", "")
            raw_impact = target_obj.get("impact")
            raw_evidence = target_obj.get("evidence")
            raw_actions = target_obj.get("actions_taken")
            if raw_actions is None:
                raw_actions = target_obj.get("actions")
        elif hasattr(target_obj, "title") and (description is None or not isinstance(title, str)):
            raw_title = getattr(target_obj, "title", "")
            raw_desc = getattr(target_obj, "description", "")
            raw_impact = getattr(target_obj, "impact", None)
            raw_evidence = getattr(target_obj, "evidence", None)
            raw_actions = getattr(target_obj, "actions_taken", None)
            if raw_actions is None:
                raw_actions = getattr(target_obj, "actions", None)
        else:
            raw_title = str(title) if title is not None else ""
            raw_desc = str(description) if description is not None else ""
            raw_impact = impact
            raw_evidence = evidence
            raw_actions = actions_taken if actions_taken is not None else kwargs.get("actions")

        # Step 1: Calculate provided_fields_count (count non-empty strings among impact, evidence, actions_taken)
        provided_fields_count = sum(
            1
            for field in (raw_impact, raw_evidence, raw_actions)
            if isinstance(field, str) and field.strip()
        )

        # Step 2: Generate sanitized prompts via format_assessment_prompt
        system_prompt, user_prompt = format_assessment_prompt(
            title=raw_title,
            description=raw_desc,
            impact=raw_impact,
            evidence=raw_evidence,
            actions=raw_actions,
        )

        # Step 3: Request semantic perception from LLM client
        assessment: LLMAssessment = await self.llm_client.complete_assessment(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Step 4: Handle invalid / non-incident report input
        if not assessment.is_valid_incident_report:
            invalid_dimensions = [
                {
                    "dimension": dim.value,
                    "score": (
                        assessment.dimensions[dim].score
                        if assessment.dimensions and dim in assessment.dimensions
                        else 0
                    ),
                    "weight": QUALITY_WEIGHTS[dim],
                    "justification": (
                        assessment.dimensions[dim].justification
                        if assessment.dimensions and dim in assessment.dimensions
                        else "Not a valid incident report."
                    ),
                    "confidence": (
                        assessment.dimensions[dim].confidence
                        if assessment.dimensions and dim in assessment.dimensions
                        else 0.0
                    ),
                }
                for dim in QualityDimension
            ]

            return AssessmentResult(
                status="not_an_incident_report",
                report_quality=AttrDict({
                    "score": 0.0,
                    "band": "POOR",
                    "dimensions": invalid_dimensions,
                }),
                severity=AttrDict({
                    "level": SeverityLevel.SEV4,
                    "index": 0.0,
                    "applied_rules": [],
                }),
                confidence=AttrDict({
                    "value": MIN_CONFIDENCE,
                    "band": BAND_LOW,
                    "limiting_reason": LIMITING_REASON_WEAK_EVIDENCE,
                    "factors": {
                        "model_self_reported": 0.0,
                        "evidence_ceiling": 0.30,
                        "coverage_factor": 0.75,
                        "gap_penalty": 0.0,
                    },
                }),
                missing_information=assessment.missing_information,
                missing_information_notes=assessment.missing_information_notes,
                summary_explanation=assessment.summary,
                requires_human_review=True,
            )

        # Step 5: Calculate deterministic quality score and band
        quality_score, quality_band = compute_quality_score(assessment.dimensions)

        # Step 6: Calculate deterministic severity and policy floors
        sev_level, sev_index, applied_floors = compute_severity(assessment.severity_signals)

        # Step 7: Calculate calibrated confidence
        conf_dict = compute_confidence(
            assessment=assessment,
            provided_fields_count=provided_fields_count,
            total_optional_fields=3,
        )

        # Step 8: Determine requires_human_review
        requires_human_review = bool(
            conf_dict.get("band") == "LOW"
            or quality_band == "POOR"
            or len(applied_floors) > 0
            or sev_level == SeverityLevel.SEV1
        )

        # Step 9: Build and return AssessmentResult with complete dimension breakdowns
        dimension_breakdowns = [
            {
                "dimension": dim.value,
                "score": assessment.dimensions[dim].score,
                "weight": QUALITY_WEIGHTS[dim],
                "justification": assessment.dimensions[dim].justification,
                "confidence": assessment.dimensions[dim].confidence,
            }
            for dim in QualityDimension
        ]

        return AssessmentResult(
            status="assessed",
            report_quality=AttrDict({
                "score": quality_score,
                "band": quality_band,
                "dimensions": dimension_breakdowns,
            }),
            severity=AttrDict({
                "level": sev_level,
                "index": sev_index,
                "applied_rules": applied_floors,
            }),
            confidence=AttrDict(conf_dict),
            missing_information=assessment.missing_information,
            missing_information_notes=assessment.missing_information_notes,
            summary_explanation=assessment.summary,
            requires_human_review=requires_human_review,
        )
