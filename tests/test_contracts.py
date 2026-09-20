import pytest
from pydantic import ValidationError

from app.judgment.contracts import (
    DimensionSignal,
    LLMAssessment,
    SeveritySignals,
)
from app.judgment.rubric import MissingInfoCode, QualityDimension


def make_valid_dimension_signal_data(
    justification: str = "Clear explanation of impact and steps.",
    evidence_quote: str | None = "Error rate jumped to 45%.",
    score: int = 4,
    confidence: float = 0.9,
    **extra,
) -> dict:
    data = {
        "justification": justification,
        "evidence_quote": evidence_quote,
        "score": score,
        "confidence": confidence,
    }
    data.update(extra)
    return data


def make_valid_severity_signals_data(
    reasoning: str = "Production auth service outage affecting subset of regions.",
    impact_scope: int = 2,
    business_criticality: int = 3,
    time_sensitivity: int = 2,
    confidence: float = 0.95,
    **extra,
) -> dict:
    data = {
        "reasoning": reasoning,
        "impact_scope": impact_scope,
        "business_criticality": business_criticality,
        "time_sensitivity": time_sensitivity,
        "confidence": confidence,
    }
    data.update(extra)
    return data


def make_valid_llm_assessment_data(**overrides) -> dict:
    dimensions_data = {
        dim.value: make_valid_dimension_signal_data()
        for dim in QualityDimension
    }
    base = {
        "is_valid_incident_report": True,
        "dimensions": dimensions_data,
        "severity_signals": make_valid_severity_signals_data(),
        "missing_information": [MissingInfoCode.NO_TIMELINE],
        "missing_information_notes": "Specific outage start timestamp was not provided.",
        "summary": "Authentication failure causing elevated HTTP 500 errors in US-East region.",
    }
    base.update(overrides)
    return base


# 1. Valid full assessment parses cleanly
def test_valid_full_assessment_parses_cleanly():
    payload = make_valid_llm_assessment_data()
    assessment = LLMAssessment.model_validate(payload)

    assert assessment.is_valid_incident_report is True
    assert len(assessment.dimensions) == len(QualityDimension)
    for dim in QualityDimension:
        assert dim in assessment.dimensions
        assert assessment.dimensions[dim].score == 4
        assert assessment.dimensions[dim].confidence == 0.9
        assert assessment.dimensions[dim].justification == "Clear explanation of impact and steps."
        assert assessment.dimensions[dim].evidence_quote == "Error rate jumped to 45%."

    assert assessment.severity_signals.impact_scope == 2
    assert assessment.severity_signals.business_criticality == 3
    assert assessment.severity_signals.time_sensitivity == 2
    assert assessment.severity_signals.confidence == 0.95
    assert assessment.missing_information == [MissingInfoCode.NO_TIMELINE]
    assert assessment.missing_information_notes == "Specific outage start timestamp was not provided."
    assert assessment.summary == "Authentication failure causing elevated HTTP 500 errors in US-East region."


def test_valid_assessment_with_defaults():
    payload = make_valid_llm_assessment_data()
    del payload["missing_information"]
    del payload["missing_information_notes"]
    # Also test evidence_quote default None
    payload["dimensions"][QualityDimension.CLARITY.value]["evidence_quote"] = None

    assessment = LLMAssessment.model_validate(payload)
    assert assessment.missing_information == []
    assert assessment.missing_information_notes == ""
    assert assessment.dimensions[QualityDimension.CLARITY].evidence_quote is None


def test_invalid_incident_report_flag():
    payload = make_valid_llm_assessment_data(is_valid_incident_report=False)
    assessment = LLMAssessment.model_validate(payload)
    assert assessment.is_valid_incident_report is False


# 2. Missing dimension in dimensions dictionary raises ValidationError
def test_missing_single_dimension_raises_validation_error():
    payload = make_valid_llm_assessment_data()
    del payload["dimensions"][QualityDimension.ACTION_CONTEXT.value]

    with pytest.raises(ValidationError) as exc_info:
        LLMAssessment.model_validate(payload)

    error_str = str(exc_info.value)
    assert "Model failed to supply dimensions: ['ACTION_CONTEXT']" in error_str


def test_missing_multiple_dimensions_raises_validation_error():
    payload = make_valid_llm_assessment_data()
    del payload["dimensions"][QualityDimension.CLARITY.value]
    del payload["dimensions"][QualityDimension.REPRODUCIBILITY.value]

    with pytest.raises(ValidationError) as exc_info:
        LLMAssessment.model_validate(payload)

    error_str = str(exc_info.value)
    assert "Model failed to supply dimensions: ['CLARITY', 'REPRODUCIBILITY']" in error_str


def test_empty_dimensions_raises_validation_error():
    payload = make_valid_llm_assessment_data(dimensions={})

    with pytest.raises(ValidationError) as exc_info:
        LLMAssessment.model_validate(payload)

    expected_all = sorted(d.value for d in QualityDimension)
    assert f"Model failed to supply dimensions: {expected_all}" in str(exc_info.value)


# 3. Out-of-bounds score raises ValidationError
@pytest.mark.parametrize("invalid_score", [-1, -5, 6, 10])
def test_out_of_bounds_dimension_score_raises_validation_error(invalid_score):
    payload = make_valid_dimension_signal_data(score=invalid_score)
    with pytest.raises(ValidationError):
        DimensionSignal.model_validate(payload)


@pytest.mark.parametrize("valid_score", [0, 1, 2, 3, 4, 5])
def test_boundary_dimension_scores_valid(valid_score):
    payload = make_valid_dimension_signal_data(score=valid_score)
    signal = DimensionSignal.model_validate(payload)
    assert signal.score == valid_score


# 4. Out-of-bounds confidence raises ValidationError
@pytest.mark.parametrize("invalid_confidence", [-0.1, -1.0, 1.01, 1.5, 2.0])
def test_out_of_bounds_dimension_confidence_raises_validation_error(invalid_confidence):
    payload = make_valid_dimension_signal_data(confidence=invalid_confidence)
    with pytest.raises(ValidationError):
        DimensionSignal.model_validate(payload)


@pytest.mark.parametrize("valid_confidence", [0.0, 0.5, 1.0])
def test_boundary_dimension_confidence_valid(valid_confidence):
    payload = make_valid_dimension_signal_data(confidence=valid_confidence)
    signal = DimensionSignal.model_validate(payload)
    assert signal.confidence == valid_confidence


@pytest.mark.parametrize("invalid_confidence", [-0.01, 1.001, 2.0])
def test_out_of_bounds_severity_confidence_raises_validation_error(invalid_confidence):
    payload = make_valid_severity_signals_data(confidence=invalid_confidence)
    with pytest.raises(ValidationError):
        SeveritySignals.model_validate(payload)


# Out-of-bounds severity ordinal fields
@pytest.mark.parametrize("invalid_scope", [-1, 5, 10])
def test_out_of_bounds_impact_scope_raises_validation_error(invalid_scope):
    payload = make_valid_severity_signals_data(impact_scope=invalid_scope)
    with pytest.raises(ValidationError):
        SeveritySignals.model_validate(payload)


@pytest.mark.parametrize("valid_scope", [0, 1, 2, 3, 4])
def test_boundary_impact_scope_valid(valid_scope):
    payload = make_valid_severity_signals_data(impact_scope=valid_scope)
    signals = SeveritySignals.model_validate(payload)
    assert signals.impact_scope == valid_scope


@pytest.mark.parametrize("invalid_crit", [-1, 5, 10])
def test_out_of_bounds_business_criticality_raises_validation_error(invalid_crit):
    payload = make_valid_severity_signals_data(business_criticality=invalid_crit)
    with pytest.raises(ValidationError):
        SeveritySignals.model_validate(payload)


@pytest.mark.parametrize("valid_crit", [0, 1, 2, 3, 4])
def test_boundary_business_criticality_valid(valid_crit):
    payload = make_valid_severity_signals_data(business_criticality=valid_crit)
    signals = SeveritySignals.model_validate(payload)
    assert signals.business_criticality == valid_crit


@pytest.mark.parametrize("invalid_time", [-1, 4, 10])
def test_out_of_bounds_time_sensitivity_raises_validation_error(invalid_time):
    payload = make_valid_severity_signals_data(time_sensitivity=invalid_time)
    with pytest.raises(ValidationError):
        SeveritySignals.model_validate(payload)


@pytest.mark.parametrize("valid_time", [0, 1, 2, 3])
def test_boundary_time_sensitivity_valid(valid_time):
    payload = make_valid_severity_signals_data(time_sensitivity=valid_time)
    signals = SeveritySignals.model_validate(payload)
    assert signals.time_sensitivity == valid_time


# 5. Extra forbidden fields raise ValidationError
def test_extra_forbidden_fields_in_dimension_signal():
    payload = make_valid_dimension_signal_data(unexpected_field="disallowed")
    with pytest.raises(ValidationError) as exc_info:
        DimensionSignal.model_validate(payload)
    assert "extra_forbidden" in str(exc_info.value)


def test_extra_forbidden_fields_in_severity_signals():
    payload = make_valid_severity_signals_data(extra_metric=99)
    with pytest.raises(ValidationError) as exc_info:
        SeveritySignals.model_validate(payload)
    assert "extra_forbidden" in str(exc_info.value)


def test_extra_forbidden_fields_in_llm_assessment():
    payload = make_valid_llm_assessment_data(spurious_injection="attack_payload")
    with pytest.raises(ValidationError) as exc_info:
        LLMAssessment.model_validate(payload)
    assert "extra_forbidden" in str(exc_info.value)


# Max length validations
def test_max_length_justification_in_dimension_signal():
    payload = make_valid_dimension_signal_data(justification="a" * 401)
    with pytest.raises(ValidationError):
        DimensionSignal.model_validate(payload)

    # 400 characters is allowed
    payload_valid = make_valid_dimension_signal_data(justification="a" * 400)
    signal = DimensionSignal.model_validate(payload_valid)
    assert len(signal.justification) == 400


def test_max_length_evidence_quote_in_dimension_signal():
    payload = make_valid_dimension_signal_data(evidence_quote="q" * 201)
    with pytest.raises(ValidationError):
        DimensionSignal.model_validate(payload)

    payload_valid = make_valid_dimension_signal_data(evidence_quote="q" * 200)
    signal = DimensionSignal.model_validate(payload_valid)
    assert len(signal.evidence_quote) == 200


def test_max_length_reasoning_in_severity_signals():
    payload = make_valid_severity_signals_data(reasoning="r" * 501)
    with pytest.raises(ValidationError):
        SeveritySignals.model_validate(payload)

    payload_valid = make_valid_severity_signals_data(reasoning="r" * 500)
    signals = SeveritySignals.model_validate(payload_valid)
    assert len(signals.reasoning) == 500


def test_max_length_summary_in_llm_assessment():
    payload = make_valid_llm_assessment_data(summary="s" * 601)
    with pytest.raises(ValidationError):
        LLMAssessment.model_validate(payload)

    payload_valid = make_valid_llm_assessment_data(summary="s" * 600)
    assessment = LLMAssessment.model_validate(payload_valid)
    assert len(assessment.summary) == 600


def test_max_length_missing_info_notes_in_llm_assessment():
    payload = make_valid_llm_assessment_data(missing_information_notes="n" * 501)
    with pytest.raises(ValidationError):
        LLMAssessment.model_validate(payload)

    payload_valid = make_valid_llm_assessment_data(missing_information_notes="n" * 500)
    assessment = LLMAssessment.model_validate(payload_valid)
    assert len(assessment.missing_information_notes) == 500


def test_max_items_missing_information_in_llm_assessment():
    codes = [MissingInfoCode.NO_TIMELINE] * 11
    payload = make_valid_llm_assessment_data(missing_information=codes)
    with pytest.raises(ValidationError):
        LLMAssessment.model_validate(payload)

    codes_valid = [MissingInfoCode.NO_TIMELINE] * 10
    payload_valid = make_valid_llm_assessment_data(missing_information=codes_valid)
    assessment = LLMAssessment.model_validate(payload_valid)
    assert len(assessment.missing_information) == 10


# 6. Schema JSON export works cleanly & enforces autoregressive ordering
def test_schema_json_export_and_autoregressive_ordering():
    schema = LLMAssessment.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "dimensions" in schema["properties"]
    assert "severity_signals" in schema["properties"]
    assert schema["additionalProperties"] is False

    # Check DimensionSignal schema ordering
    dim_schema = DimensionSignal.model_json_schema()
    dim_props = list(dim_schema["properties"].keys())
    assert dim_schema["additionalProperties"] is False
    assert dim_props.index("justification") < dim_props.index("score")
    assert dim_props.index("justification") < dim_props.index("confidence")

    # Check SeveritySignals schema ordering
    sev_schema = SeveritySignals.model_json_schema()
    sev_props = list(sev_schema["properties"].keys())
    assert sev_schema["additionalProperties"] is False
    assert sev_props.index("reasoning") < sev_props.index("impact_scope")
    assert sev_props.index("reasoning") < sev_props.index("business_criticality")
    assert sev_props.index("reasoning") < sev_props.index("time_sensitivity")
    assert sev_props.index("reasoning") < sev_props.index("confidence")
