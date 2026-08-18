import pytest

from app.models.mock import create_reasoning_model
from app.telemetry.cost import estimate_cost


def test_estimate_cost_uses_input_and_output_pricing() -> None:
    model = create_reasoning_model()

    cost = estimate_cost(
        metadata=model.metadata,
        input_tokens=500,
        output_tokens=1_000,
    )

    assert cost == pytest.approx(0.000675)


def test_estimate_cost_rejects_negative_token_counts() -> None:
    model = create_reasoning_model()

    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        estimate_cost(
            metadata=model.metadata,
            input_tokens=-1,
            output_tokens=100,
        )