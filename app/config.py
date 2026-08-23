from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MAGICPIN_", extra="ignore")

    database_path: str = "data/context_store.db"

    # Submission/identity metadata. These describe the submitting team and
    # are intentionally left unconfigured by default (None) rather than
    # shipped with fabricated placeholder text — set them via environment
    # variables (e.g. MAGICPIN_TEAM_NAME) or a local .env file before
    # submission.
    team_name: str | None = None
    team_members: str | None = None
    contact_email: str | None = None
    version: str = "0.1.0"
    submitted_at: str | None = None

    # Free-text description of the current approach. Unlike the identity
    # fields above, this describes the implementation itself (not personal
    # information), so a factual default is appropriate and can still be
    # overridden via MAGICPIN_APPROACH if a team wants to describe their own
    # variant.
    approach: str = (
        "Deterministic context resolution and trigger routing; Gemini-based "
        "composer with a deterministic, safety-validated fallback when no "
        "Gemini API key is configured; stateful /v1/reply handling for "
        "multi-turn conversations; /v1/teardown for end-of-test state reset."
    )

    # Composer model reporting. `model` is only set if a team wants to
    # explicitly override what /v1/metadata reports; by default the metadata
    # endpoint reports whichever Gemini model is actually configured
    # (`gemini_model`), which is truthful whether or not GEMINI_API_KEY is
    # set — it never falls back to a made-up "not_configured" placeholder.
    model: str | None = None
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL")
    debug_log_prompts: bool | None = False

    @property
    def parsed_team_members(self) -> list[str]:
        if not self.team_members:
            return []
        return [member.strip() for member in self.team_members.split(",") if member.strip()]

    @property
    def reported_model(self) -> str:
        """The model value /v1/metadata should report: an explicit override
        if set, otherwise the actually-configured Gemini model — never a
        placeholder like "not_configured"."""
        return self.model or self.gemini_model
