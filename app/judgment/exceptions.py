"""Domain exceptions for the incident judgment and scoring service.

These exceptions are framework-agnostic with zero HTTP or third-party dependencies.
"""


class IncidentJudgmentError(Exception):
    """Base domain exception for all incident judgment operations."""


class LLMProviderError(IncidentJudgmentError):
    """Raised when an external LLM provider call fails, times out, or returns an unrecoverable error."""


class SchemaValidationError(IncidentJudgmentError):
    """Raised when an LLM completion or domain input violates expected schema/format contracts."""


class UntrustedInputError(IncidentJudgmentError):
    """Raised when raw incident text fails safety, sanitization, or contains untrusted payloads."""
