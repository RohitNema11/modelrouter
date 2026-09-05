import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def use_mock_provider_for_tests(monkeypatch: pytest.MonkeyPatch):
    """Keep unit and API tests deterministic and independent of provider credentials."""
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()