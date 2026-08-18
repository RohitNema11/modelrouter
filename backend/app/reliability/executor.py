from dataclasses import dataclass

from app.models.base import BaseModel, ModelResponse
from app.models.registry import ModelRegistry


@dataclass(frozen=True)
class ExecutionResult:
    """A successful model result plus reliability information."""

    response: ModelResponse
    initial_model: str
    fallback_used: bool
    attempts: int


class ModelExecutor:
    """Generate with retry and higher-tier fallback support."""

    _TIER_RANK = {"fast": 0, "general": 1, "reasoning": 2}

    def __init__(self, registry: ModelRegistry, max_retries: int) -> None:
        self._registry = registry
        self._max_retries = max_retries

    async def generate(
        self,
        selected_model_name: str,
        prompt: str,
        required_capabilities: list[str],
    ) -> ExecutionResult:
        initial_model = self._registry.get(selected_model_name)
        candidates = [initial_model] + self._fallback_candidates(
            initial_model=initial_model,
            required_capabilities=required_capabilities,
        )

        attempts = 0
        last_error: Exception | None = None

        for candidate in candidates:
            for _ in range(self._max_retries + 1):
                attempts += 1

                try:
                    response = await candidate.generate(prompt)
                    return ExecutionResult(
                        response=response,
                        initial_model=initial_model.metadata.name,
                        fallback_used=(
                            candidate.metadata.name != initial_model.metadata.name
                        ),
                        attempts=attempts,
                    )
                except Exception as error:
                    last_error = error

        raise RuntimeError(
            f"All generation attempts failed after {attempts} attempts."
        ) from last_error

    def _fallback_candidates(
        self,
        initial_model: BaseModel,
        required_capabilities: list[str],
    ) -> list[BaseModel]:
        initial_rank = self._TIER_RANK[initial_model.metadata.tier]

        candidates = [
            model
            for model in self._registry.list_models()
            if self._TIER_RANK[model.metadata.tier] > initial_rank
            and set(required_capabilities).issubset(model.metadata.capabilities)
        ]

        return sorted(
            candidates,
            key=lambda model: self._TIER_RANK[model.metadata.tier],
        )