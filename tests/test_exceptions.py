import ast
from pathlib import Path

import pytest

from app.judgment.exceptions import (
    IncidentJudgmentError,
    LLMProviderError,
    SchemaValidationError,
    UntrustedInputError,
)


def test_exception_hierarchy():
    # Verify inheritance
    assert issubclass(LLMProviderError, IncidentJudgmentError)
    assert issubclass(SchemaValidationError, IncidentJudgmentError)
    assert issubclass(UntrustedInputError, IncidentJudgmentError)
    assert issubclass(IncidentJudgmentError, Exception)


def test_exception_catching():
    with pytest.raises(IncidentJudgmentError) as exc_info:
        raise LLMProviderError("OpenAI connection timed out")
    assert "OpenAI connection timed out" in str(exc_info.value)

    with pytest.raises(IncidentJudgmentError) as exc_info:
        raise SchemaValidationError("Invalid JSON output")
    assert "Invalid JSON output" in str(exc_info.value)

    with pytest.raises(IncidentJudgmentError) as exc_info:
        raise UntrustedInputError("Prompt injection attempt detected")
    assert "Prompt injection attempt detected" in str(exc_info.value)


def test_domain_zero_framework_dependencies():
    """Verify that app/judgment has ZERO framework (fastapi, starlette) or HTTP dependencies."""
    forbidden_modules = {
        "fastapi",
        "starlette",
        "httpx",
        "requests",
        "urllib.request",
        "aiohttp",
        "flask",
        "uvicorn",
    }

    judgment_dir = Path(__file__).resolve().parent.parent / "app" / "judgment"
    python_files = list(judgment_dir.glob("*.py"))

    assert len(python_files) > 0, "No python files found in app/judgment"

    for file_path in python_files:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert (
                        root_mod not in forbidden_modules
                    ), f"Forbidden import '{alias.name}' found in {file_path.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    assert (
                        root_mod not in forbidden_modules
                    ), f"Forbidden from-import '{node.module}' found in {file_path.name}"
