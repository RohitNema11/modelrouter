import asyncio

import pytest

from app.models.base import BaseModel, ModelMetadata, ModelResponse
from app.models.registry import ModelRegistry
from app.reliability.executor import ModelExecutor, UpstreamGenerationError


class FailingModel(BaseModel):
    """Test-only model that always fails and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.metadata = ModelMetadata(
            name="fast",
            provider="test",
            tier="fast",
            input_cost_per_million=0.05,
            output_cost_per_million=0.08,
            context_window=131_072,
            capabilities=frozenset({"general"}),
            reliability_score=0.0,
            expected_latency_ms=100,
        )

    async def generate(self, prompt: str) -> ModelResponse:
        self.calls += 1
        raise TimeoutError("Simulated timeout")


class SuccessfulModel(BaseModel):
    """Test-only model that returns a deterministic response."""

    def __init__(self, name: str, tier: str) -> None:
        self.metadata = ModelMetadata(
            name=name,
            provider="test",
            tier=tier,
            input_cost_per_million=0.05,
            output_cost_per_million=0.08,
            context_window=131_072,
            capabilities=frozenset({"general"}),
            reliability_score=1.0,
            expected_latency_ms=100,
        )

    async def generate(self, prompt: str) -> ModelResponse:
        return ModelResponse(
            content="Success",
            input_tokens=1,
            output_tokens=1,
            model_name=self.metadata.name,
        )


def test_executor_returns_first_model_when_it_succeeds() -> None:
    registry = ModelRegistry(models=[SuccessfulModel("fast", "fast")])
    executor = ModelExecutor(registry=registry)

    result = asyncio.run(
        executor.generate(
            selected_model_name="fast",
            prompt="What is the capital of Japan?",
            required_capabilities=["general"],
        )
    )

    assert result.response.model_name == "fast"
    assert result.initial_model == "fast"
    assert result.fallback_used is False
    assert result.attempts == 1


def test_executor_falls_back_to_a_compatible_higher_tier_after_timeout() -> None:
    failing_fast_model = FailingModel()
    general_model = SuccessfulModel("general", "general")
    registry = ModelRegistry(
        models=[failing_fast_model, general_model]
    )
    executor = ModelExecutor(
        registry=registry,
        max_retries=0,
        retry_backoff_seconds=0,
    )

    result = asyncio.run(
        executor.generate(
            selected_model_name="fast",
            prompt="What is the capital of Japan?",
            required_capabilities=["general"],
        )
    )

    assert failing_fast_model.calls == 1
    assert result.response.model_name == "general"
    assert result.initial_model == "fast"
    assert result.fallback_used is True
    assert result.attempts == 2


def test_executor_reports_a_normalized_error_after_retryable_failure(caplog) -> None:
    failing_fast_model = FailingModel()
    registry = ModelRegistry(models=[failing_fast_model])
    executor = ModelExecutor(registry=registry, max_retries=1)

    with pytest.raises(UpstreamGenerationError, match="timeout") as error:
        asyncio.run(
            executor.generate(
                selected_model_name="fast",
                prompt="What is the capital of Japan?",
                required_capabilities=["general"],
            )
        )

    assert error.value.reason == "timeout"
    assert error.value.attempts == 2
    assert failing_fast_model.calls == 2
    assert "Generation attempt failed: model=fast" in caplog.text
    assert "Simulated timeout" in caplog.text
