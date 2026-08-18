import pytest
import asyncio

from app.models.mock import create_fast_model
from app.models.registry import create_model_registry
from app.core.config import get_settings


def test_registry_contains_all_default_models() -> None:
    registry = create_model_registry()

    assert [model.metadata.name for model in registry.list_models()] == [
        "fast",
        "general",
        "reasoning",
    ]


def test_registry_returns_model_by_name() -> None:
    registry = create_model_registry()

    model = registry.get("reasoning")

    assert model.metadata.tier == "reasoning"
    assert "complex_reasoning" in model.metadata.capabilities


def test_registry_rejects_unknown_model() -> None:
    registry = create_model_registry()

    with pytest.raises(ValueError, match="Unknown model: missing"):
        registry.get("missing")


def test_mock_model_returns_normalized_response() -> None:
    model = create_fast_model()

    response = asyncio.run(model.generate("Summarize this short text."))

    assert response.model_name == "fast"
    assert response.input_tokens == 4
    assert response.output_tokens > 0
    assert response.content.startswith("[Mock fast response]")

def test_registry_creates_groq_models_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    try:
        registry = create_model_registry()

        assert [model.metadata.name for model in registry.list_models()] == [
            "fast",
            "general",
            "reasoning",
        ]
        assert all(
            model.metadata.provider == "groq"
            for model in registry.list_models()
        )
    finally:
        get_settings.cache_clear()