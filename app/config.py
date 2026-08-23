from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MAGICPIN_",
        extra="ignore",
    )

    database_path: str = "data/context_store.db"

    team_name: str | None = None
    team_members: str | None = None
    contact_email: str | None = None

    version: str = "0.1.0"
    submitted_at: str | None = None

    approach: str = (
        "Deterministic context resolution and trigger routing; Gemini-based "
        "composer with a deterministic, safety-validated fallback when no "
        "Gemini API key is configured; stateful /v1/reply handling for "
        "multi-turn conversations; /v1/teardown for end-of-test state reset."
    )

    model: str | None = None

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )

    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias="GEMINI_MODEL",
    )

    debug_log_prompts: bool = False

    @field_validator("debug_log_prompts", mode="before")
    @classmethod
    def handle_empty_debug_log_prompts(cls, value):
        if value is None or value == "":
            return False

        if isinstance(value, str):
            value = value.strip().lower()

            if value in {"true", "1", "yes", "on"}:
                return True

            if value in {"false", "0", "no", "off"}:
                return False

        return value

    @property
    def parsed_team_members(self) -> list[str]:
        if not self.team_members:
            return []

        return [
            member.strip()
            for member in self.team_members.split(",")
            if member.strip()
        ]

    @property
    def reported_model(self) -> str:
        """Return the configured model name."""
        return self.model or self.gemini_model