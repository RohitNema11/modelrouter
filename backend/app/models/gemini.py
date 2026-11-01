from google import genai
from google.genai import types

from app.models.base import BaseModel, ModelMetadata, ModelResponse


class GeminiModel(BaseModel):
    """Gemini-backed implementation of the shared model interface."""

    def __init__(
        self,
        metadata: ModelMetadata,
        gemini_model_id: str,
        api_key: str,
        timeout_seconds: float,
    ) -> None:
        self.metadata = metadata
        self.gemini_model_id = gemini_model_id
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1_000),
            ),
        )

    async def generate(self, prompt: str) -> ModelResponse:
        """Generate a response through Gemini and normalize its result."""
        response = await self._client.aio.models.generate_content(
            model=self.gemini_model_id,
            contents=prompt,
        )
        usage = response.usage_metadata

        return ModelResponse(
            content=response.text or "",
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            model_name=self.metadata.name,
        )


def create_gemini_models(
    api_key: str,
    timeout_seconds: float,
    reasoning_timeout_seconds: float,
) -> list[GeminiModel]:
    """Create the three configured Gemini model tiers."""
    model_definitions = [
        ModelMetadata(
            name="fast",
            provider="gemini",
            tier="fast",
            model_id="gemini-3.5-flash-lite",
            input_cost_per_million=0.30,
            output_cost_per_million=2.50,
            context_window=131_072,
            capabilities=frozenset(
                {"general", "summarization", "extraction", "classification"}
            ),
            reliability_score=0.99,
            expected_latency_ms=100,
        ),
        ModelMetadata(
            name="general",
            provider="gemini",
            tier="general",
            model_id="gemini-3.6-flash",
            input_cost_per_million=0.75,
            output_cost_per_million=3.75,
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
            provider="gemini",
            tier="reasoning",
            model_id="gemini-3.8-flash",
            input_cost_per_million=0.75,
            output_cost_per_million=3.75,
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

    return [
        GeminiModel(
            metadata=metadata,
            gemini_model_id=metadata.model_id or metadata.name,
            api_key=api_key,
            timeout_seconds=(
                reasoning_timeout_seconds
                if metadata.tier == "reasoning"
                else timeout_seconds
            ),
        )
        for metadata in model_definitions
    ]
