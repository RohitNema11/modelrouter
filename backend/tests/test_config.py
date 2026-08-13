from app.core.config import get_settings


def test_default_settings() -> None:
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.app_name == "ModelRouter API"
    assert settings.environment == "development"
    assert settings.debug is True