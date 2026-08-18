from app.core.config import get_settings


def test_default_settings() -> None:
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.app_name == "ModelRouter API"
    assert settings.environment == "development"
    assert settings.debug is True
    assert settings.groq_api_key is None
    assert settings.request_timeout_seconds == 30.0
    assert settings.max_retries == 1
    assert settings.model_provider == "mock"