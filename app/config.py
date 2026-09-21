"""Application Configuration and Environment Settings.

Uses Pydantic Settings V2 to load configuration from environment variables
and optional .env file with strict typing and sensible production defaults.
"""

import json
from typing import Any
from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class CustomEnvSettingsSource(EnvSettingsSource):
    """Custom environment settings source to gracefully handle comma-separated lists."""

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        if field_name == "cors_origins" and isinstance(value, str):
            val = value.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in val.split(",") if item.strip()]
        return super().decode_complex_value(field_name, field, value)


class CustomDotEnvSettingsSource(DotEnvSettingsSource):
    """Custom dotenv settings source to gracefully handle comma-separated lists."""

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        if field_name == "cors_origins" and isinstance(value, str):
            val = value.strip()
            if val.startswith("[") and val.endswith("]"):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in val.split(",") if item.strip()]
        return super().decode_complex_value(field_name, field, value)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and/or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Incident Judgment & Scoring Service"
    environment: str = "development"
    llm_provider: str = "fake"  # options: "fake", "openai", "gemini"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 30.0
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: float = 30.0
    cors_origins: list[str] = ["*"]

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate that llm_provider is one of the supported options."""
        cleaned = v.strip().lower()
        if cleaned not in {"fake", "openai", "gemini"}:
            raise ValueError(
                f"Invalid llm_provider: '{v}'. Supported options: 'fake', 'openai', 'gemini'"
            )
        return cleaned

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str]:
        """Support comma-separated strings or JSON arrays for CORS origins."""
        if isinstance(v, str):
            cleaned = v.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except json.JSONDecodeError:
                    pass
            return [i.strip() for i in cleaned.split(",") if i.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v]
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize sources to support both comma-separated and JSON formats for lists."""
        return (
            init_settings,
            CustomEnvSettingsSource(settings_cls),
            CustomDotEnvSettingsSource(settings_cls),
            file_secret_settings,
        )


# Global singleton settings instance
settings = Settings()
