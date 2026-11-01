from app.models.base import ModelMetadata


def estimate_cost(
    metadata: ModelMetadata,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate on-demand API cost in USD from normalized token usage."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    if (
        metadata.input_cost_per_million is None
        or metadata.output_cost_per_million is None
    ):
        return None

    input_cost = (
        input_tokens / 1_000_000
    ) * metadata.input_cost_per_million
    output_cost = (
        output_tokens / 1_000_000
    ) * metadata.output_cost_per_million

    return input_cost + output_cost