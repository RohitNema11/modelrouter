from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ModelMetadata:
    """Static information used by the router to compare models."""

    name: str
    provider: str
    tier: str
    input_cost_per_million: float | None
    output_cost_per_million: float | None
    context_window: int
    capabilities: FrozenSet[str]
    reliability_score: float
    expected_latency_ms: int
    model_id: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    """Provider-independent result returned after model generation."""

    content: str
    input_tokens: int
    output_tokens: int
    model_name: str


class BaseModel(ABC):
    """Common interface that every model adapter must implement."""

    metadata: ModelMetadata

    @abstractmethod
    async def generate(self, prompt: str) -> ModelResponse:
        """Generate a response for a user prompt."""