from app.core.config import get_settings


def test_default_settings(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.app_name == "ModelRouter API"
    assert settings.environment == "development"
    assert settings.gemini_api_key is None
    assert settings.request_timeout_seconds == 60.0
    assert settings.reasoning_timeout_seconds == 400.0
    assert settings.max_retries == 1
    assert settings.retry_backoff_seconds == 0.25
