import asyncio
import logging
from dataclasses import dataclass

import httpx

from app.models.base import BaseModel, ModelResponse
from app.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    """A successful model result plus reliability information."""

    response: ModelResponse
    initial_model: str
    fallback_used: bool
    attempts: int


class UpstreamGenerationError(RuntimeError):
    """A normalized provider failure that is safe to return to API clients."""

    def __init__(self, reason: str, attempts: int) -> None:
        super().__init__(f"Model generation failed after {attempts} attempt(s): {reason}")
        self.reason = reason
        self.attempts = attempts


class ModelExecutor:
    """Generate with bounded retries and capability-safe higher-tier fallback."""

    _TIER_RANK = {"fast": 0, "general": 1, "reasoning": 2}

    def __init__(
        self,
        registry: ModelRegistry,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        self._registry = registry
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    async def generate(
        self,
        selected_model_name: str,
        prompt: str,
        required_capabilities: list[str],
        allow_fallback: bool = True,
    ) -> ExecutionResult:
        initial_model = self._registry.get(selected_model_name)
        candidates = [initial_model]
        if allow_fallback:
            candidates.extend(
                self._fallback_candidates(
                    initial_model=initial_model,
                    required_capabilities=required_capabilities,
                )
            )

        attempts = 0
        last_error: Exception | None = None

        for candidate_index, candidate in enumerate(candidates):
            for retry_number in range(self._max_retries + 1):
                attempts += 1
                try:
                    response = await candidate.generate(prompt)
                    return ExecutionResult(
                        response=response,
                        initial_model=initial_model.metadata.name,
                        fallback_used=candidate_index > 0,
                        attempts=attempts,
                    )
                except Exception as error:
                    last_error = error
                    reason = self._failure_reason(error)
                    retryable = self._is_retryable(error)
                    logger.warning(
                        "Generation attempt failed: model=%s attempt=%s reason=%s retryable=%s",
                        candidate.metadata.name,
                        attempts,
                        reason,
                        retryable,
                        exc_info=True,
                    )
                    if not retryable:
                        raise UpstreamGenerationError(reason, attempts) from error

                    if retry_number < self._max_retries:
                        await asyncio.sleep(
                            self._retry_backoff_seconds * (2**retry_number)
                        )

            logger.info(
                "Trying compatible fallback after transient provider failure: %s",
                candidate.metadata.name,
            )

        assert last_error is not None
        raise UpstreamGenerationError(
            self._failure_reason(last_error),
            attempts,
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

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
            return True
        if isinstance(error, httpx.RequestError):
            response = getattr(error, "response", None)
            status_code = getattr(response, "status_code", None)
            return status_code is None or status_code == 429 or status_code >= 500

        status_code = getattr(error, "status_code", None)
        return isinstance(status_code, int) and (
            status_code == 408 or status_code == 429 or status_code >= 500
        )

    @staticmethod
    def _failure_reason(error: Exception) -> str:
        if isinstance(error, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
            return "timeout"

        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return "rate_limited"
        if isinstance(status_code, int) and status_code >= 500:
            return "upstream_server_error"
        if isinstance(error, httpx.RequestError):
            return "network_error"
        return "provider_error"
