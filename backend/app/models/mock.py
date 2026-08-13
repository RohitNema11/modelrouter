from app.models.base import BaseModel, ModelMetadata, ModelResponse


class MockModel(BaseModel):
    """Local stand-in for an LLM while routing is under development."""

    def __init__(self, metadata: ModelMetadata) -> None:
        self.metadata = metadata

    async def generate(self, prompt: str) -> ModelResponse:
        """Return a predictable fake response without calling any provider."""
        input_tokens = len(prompt.split())
        content = f"[Mock {self.metadata.tier} response] {prompt}"

        return ModelResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=len(content.split()),
            model_name=self.metadata.name,
        )


def create_fast_model() -> MockModel:
    return MockModel(
        metadata=ModelMetadata(
            name="fast",
            provider="mock",
            tier="fast",
            input_cost_per_million=0.05,
            output_cost_per_million=0.08,
            context_window=131_072,
            capabilities=frozenset(
                {"general", "summarization", "extraction", "classification"}
            ),
            reliability_score=0.99,
            expected_latency_ms=100,
        )
    )


def create_general_model() -> MockModel:
    return MockModel(
        metadata=ModelMetadata(
            name="general",
            provider="mock",
            tier="general",
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
            context_window=131_072,
            capabilities=frozenset(
                {
                    "general",
                    "summarization",
                    "extraction",
                    "classification",
                    "coding",
                    "debugging",
                }
            ),
            reliability_score=0.98,
            expected_latency_ms=200,
        )
    )


def create_reasoning_model() -> MockModel:
    return MockModel(
        metadata=ModelMetadata(
            name="reasoning",
            provider="mock",
            tier="reasoning",
            input_cost_per_million=0.15,
            output_cost_per_million=0.60,
            context_window=131_072,
            capabilities=frozenset(
                {
                    "general",
                    "summarization",
                    "extraction",
                    "classification",
                    "coding",
                    "debugging",
                    "mathematics",
                    "complex_reasoning",
                }
            ),
            reliability_score=0.97,
            expected_latency_ms=350,
        )
    )