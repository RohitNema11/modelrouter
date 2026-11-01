from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ModelRouter API"
    environment: str = "development"

    gemini_api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    reasoning_timeout_seconds: float = Field(default=400.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=3)
    retry_backoff_seconds: float = Field(default=0.25, ge=0, le=10)

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def treat_blank_api_key_as_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
