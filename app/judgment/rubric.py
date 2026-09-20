from enum import Enum, IntEnum
import math

RUBRIC_VERSION: str = "1.0.0"
PROMPT_VERSION: str = "v1_structured"


class QualityDimension(str, Enum):
    CLARITY = "CLARITY"
    EVIDENCE_STRENGTH = "EVIDENCE_STRENGTH"
    IMPACT_ARTICULATION = "IMPACT_ARTICULATION"
    REPRODUCIBILITY = "REPRODUCIBILITY"
    ACTION_CONTEXT = "ACTION_CONTEXT"


QUALITY_WEIGHTS: dict[QualityDimension, float] = {
    QualityDimension.CLARITY: 0.20,
    QualityDimension.EVIDENCE_STRENGTH: 0.30,
    QualityDimension.IMPACT_ARTICULATION: 0.25,
    QualityDimension.REPRODUCIBILITY: 0.10,
    QualityDimension.ACTION_CONTEXT: 0.15,
}

# Runtime verification that quality weights strictly sum to 1.0 within tolerance
if not math.isclose(sum(QUALITY_WEIGHTS.values()), 1.0, abs_tol=1e-6):
    raise ValueError(
        f"QUALITY_WEIGHTS must strictly sum to 1.0, got {sum(QUALITY_WEIGHTS.values())}"
    )


class ImpactScope(IntEnum):
    NONE = 0
    SINGLE_USER = 1
    SMALL_SUBSET = 2
    LARGE_SUBSET = 3
    ALL_USERS = 4


class BusinessCriticality(IntEnum):
    COSMETIC = 0
    DEGRADED_UX = 1
    CORE_FLOW_IMPAIRED = 2
    REVENUE_OR_DATA_AT_RISK = 3
    DATA_LOSS_OR_BREACH = 4


class TimeSensitivity(IntEnum):
    STABLE = 0
    SLOW_DEGRADATION = 1
    ACTIVE_DEGRADATION = 2
    RAPID_ESCALATION = 3


class SeverityLevel(str, Enum):
    SEV1 = "SEV1"
    SEV2 = "SEV2"
    SEV3 = "SEV3"
    SEV4 = "SEV4"


class MissingInfoCode(str, Enum):
    IMPACT_SCOPE_UNQUANTIFIED = "IMPACT_SCOPE_UNQUANTIFIED"
    NO_TIMELINE = "NO_TIMELINE"
    NO_ERROR_DETAILS = "NO_ERROR_DETAILS"
    NO_REPRO_STEPS = "NO_REPRO_STEPS"
    AFFECTED_COMPONENT_UNKNOWN = "AFFECTED_COMPONENT_UNKNOWN"
    NO_MITIGATION_HISTORY = "NO_MITIGATION_HISTORY"
    NO_MONITORING_DATA = "NO_MONITORING_DATA"
    ENVIRONMENT_UNKNOWN = "ENVIRONMENT_UNKNOWN"
