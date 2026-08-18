import asyncio

import pytest

from app.models.base import BaseModel, ModelMetadata, ModelResponse
from app.models.mock import create_fast_model, create_general_model
from app.models.registry import ModelRegistry
from app.reliability.executor import ModelExecutor


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


def test_executor_returns_first_model_when_it_succeeds() -> None:
    registry = ModelRegistry(models=[create_fast_model(), create_general_model()])
    executor = ModelExecutor(registry=registry, max_retries=1)

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


def test_executor_retries_then_falls_back_to_higher_tier() -> None:
    failing_fast_model = FailingModel()
    registry = ModelRegistry(
        models=[failing_fast_model, create_general_model()]
    )
    executor = ModelExecutor(registry=registry, max_retries=1)

    result = asyncio.run(
        executor.generate(
            selected_model_name="fast",
            prompt="What is the capital of Japan?",
            required_capabilities=["general"],
        )
    )

    assert failing_fast_model.calls == 2
    assert result.response.model_name == "general"
    assert result.initial_model == "fast"
    assert result.fallback_used is True
    assert result.attempts == 3


def test_executor_raises_after_every_candidate_fails() -> None:
    failing_fast_model = FailingModel()
    registry = ModelRegistry(models=[failing_fast_model])
    executor = ModelExecutor(registry=registry, max_retries=0)

    with pytest.raises(RuntimeError, match="All generation attempts failed"):
        asyncio.run(
            executor.generate(
                selected_model_name="fast",
                prompt="What is the capital of Japan?",
                required_capabilities=["general"],
            )
        )