import asyncio
from types import SimpleNamespace

import pytest

from app.models.gemini import GeminiModel
from app.models.registry import create_model_registry
from app.core.config import get_settings

_generate_with_sdk = GeminiModel.generate


def test_gemini_model_normalizes_sdk_response() -> None:
    async def generate_content(**kwargs):
        assert kwargs == {
            "model": "gemini-3.5-flash-lite",
            "contents": "Summarize this.",
        }
        return SimpleNamespace(
            text="Summary.",
            usage_metadata=SimpleNamespace(
                prompt_token_count=3,
                candidates_token_count=1,
            ),
        )

    model = create_model_registry().get("fast")
    model._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )

    response = asyncio.run(_generate_with_sdk(model, "Summarize this."))

    assert response.content == "Summary."
    assert response.input_tokens == 3
    assert response.output_tokens == 1
    assert response.model_name == "fast"


def test_registry_contains_gemini_models() -> None:
    registry = create_model_registry()

    assert [model.metadata.name for model in registry.list_models()] == [
        "fast",
        "general",
        "reasoning",
    ]
    assert all(model.metadata.provider == "gemini" for model in registry.list_models())
    assert [model.metadata.model_id for model in registry.list_models()] == [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    ]
    assert [
        (
            model.metadata.input_cost_per_million,
            model.metadata.output_cost_per_million,
        )
        for model in registry.list_models()
    ] == [(0.30, 2.50), (0.75, 3.75), (0.75, 3.75)]


def test_registry_returns_model_by_name() -> None:
    registry = create_model_registry()

    model = registry.get("reasoning")

    assert model.metadata.tier == "reasoning"
    assert "complex_reasoning" in model.metadata.capabilities


def test_registry_rejects_unknown_model() -> None:
    registry = create_model_registry()

    with pytest.raises(ValueError, match="Unknown model: missing"):
        registry.get("missing")


def test_registry_creates_gemini_models(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()
    captured_timeouts: list[int] = []

    def create_client(*, api_key, http_options):
        assert api_key == "test-key"
        captured_timeouts.append(http_options.timeout)
        return SimpleNamespace()

    monkeypatch.setattr("app.models.gemini.genai.Client", create_client)

    try:
        registry = create_model_registry()

        assert [model.metadata.name for model in registry.list_models()] == [
            "fast",
            "general",
            "reasoning",
        ]
        assert all(
            model.metadata.provider == "gemini"
            for model in registry.list_models()
        )
        assert captured_timeouts == [60_000, 60_000, 400_000]
    finally:
        get_settings.cache_clear()


def test_registry_requires_gemini_api_key(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()

    try:
        with pytest.raises(ValueError, match="GEMINI_API_KEY must be configured"):
            create_model_registry()
    finally:
        get_settings.cache_clear()