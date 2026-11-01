import pytest

from app.models.base import ModelMetadata
from app.models.registry import create_model_registry
from app.telemetry.cost import estimate_cost


def create_metadata(
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> ModelMetadata:
    return ModelMetadata(
        name="test",
        provider="test",
        tier="test",
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        context_window=131_072,
        capabilities=frozenset(),
        reliability_score=1.0,
        expected_latency_ms=1,
    )


def test_estimate_cost_uses_input_and_output_pricing() -> None:
    cost = estimate_cost(
        metadata=create_metadata(0.15, 0.60),
        input_tokens=500,
        output_tokens=1_000,
    )

    assert cost == pytest.approx(0.000675)


def test_gemini_model_cost_estimates_use_registry_pricing() -> None:
    registry = create_model_registry()

    estimates = {
        model.metadata.name: estimate_cost(
            metadata=model.metadata,
            input_tokens=1_000,
            output_tokens=1_000,
        )
        for model in registry.list_models()
    }

    assert estimates == {
        "fast": pytest.approx(0.0028),
        "general": pytest.approx(0.0045),
        "reasoning": pytest.approx(0.0045),
    }


def test_estimate_cost_rejects_negative_token_counts() -> None:
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        estimate_cost(
            metadata=create_metadata(None, None),
            input_tokens=-1,
            output_tokens=100,
        )


def test_estimate_cost_is_unavailable_without_provider_pricing() -> None:
    assert estimate_cost(
        metadata=create_metadata(None, None),
        input_tokens=500,
        output_tokens=1_000,
    ) is None