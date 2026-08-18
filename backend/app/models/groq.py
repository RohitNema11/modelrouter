from groq import AsyncGroq

from app.models.base import BaseModel, ModelMetadata, ModelResponse


class GroqModel(BaseModel):
    """Groq-backed implementation of the shared model interface."""

    def __init__(
        self,
        metadata: ModelMetadata,
        groq_model_id: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.metadata = metadata
        self._groq_model_id = groq_model_id
        self._client = AsyncGroq(
            api_key=api_key,
            timeout=timeout_seconds,
        )

    async def generate(self, prompt: str) -> ModelResponse:
        """Generate a response through Groq and normalize its result."""
        completion = await self._client.chat.completions.create(
            model=self._groq_model_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = completion.choices[0].message.content or ""
        usage = completion.usage

        return ModelResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model_name=self.metadata.name,
        )

def create_groq_models(
    api_key: str,
    timeout_seconds: float,
) -> list[GroqModel]:
    """Create the three production Groq model tiers."""

    model_definitions = [
        ModelMetadata(
            name="fast",
            provider="groq",
            tier="fast",
            input_cost_per_million=0.05,
            output_cost_per_million=0.08,
            context_window=131_072,
            capabilities=frozenset(
                {"general", "summarization", "extraction", "classification"}
            ),
            reliability_score=0.99,
            expected_latency_ms=100,
        ),
        ModelMetadata(
            name="general",
            provider="groq",
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
        ),
        ModelMetadata(
            name="reasoning",
            provider="groq",
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
        ),
    ]

    groq_model_ids = {
        "fast": "llama-3.1-8b-instant",
        "general": "openai/gpt-oss-20b",
        "reasoning": "openai/gpt-oss-120b",
    }

    return [
        GroqModel(
            metadata=metadata,
            groq_model_id=groq_model_ids[metadata.name],
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        for metadata in model_definitions
    ]