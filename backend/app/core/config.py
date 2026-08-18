from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ModelRouter API"
    environment: str = "development"
    debug: bool = True
    model_provider: str = "mock"

    groq_api_key: SecretStr | None = None
    request_timeout_seconds: float = 30.0
    max_retries: int = 1

    @field_validator("groq_api_key", mode="before")
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