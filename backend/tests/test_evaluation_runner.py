import asyncio

from app.evaluation.runner import DEFAULT_CASES, run_evaluation
from app.models.base import BaseModel, ModelMetadata, ModelResponse
from app.models.registry import ModelRegistry
from app.reliability.executor import ModelExecutor
from app.schemas.routing import QueryAnalysis, RoutingDecision, TaskType
from app.services import ApplicationServices


class EvaluationModel(BaseModel):
    """Deterministic model used to test evaluation bookkeeping."""

    def __init__(self, tier: str) -> None:
        self.metadata = ModelMetadata(
            name=tier,
            provider="test",
            tier=tier,
            input_cost_per_million=1.0,
            output_cost_per_million=1.0,
            context_window=1_000,
            capabilities=frozenset({"general", "coding", "complex_reasoning"}),
            reliability_score=1.0,
            expected_latency_ms=1,
        )

    async def generate(self, prompt: str) -> ModelResponse:
        return ModelResponse(
            content=f"{self.metadata.name}: {prompt}",
            input_tokens=2,
            output_tokens=3,
            model_name=self.metadata.name,
        )


class EvaluationRouter:
    """Routes each standard case to its declared expected tier."""

    async def route(self, query: str) -> RoutingDecision:
        case = next(case for case in DEFAULT_CASES if case.query == query)
        return RoutingDecision(
            selected_model=case.expected_tier,
            confidence=1.0,
            analysis=QueryAnalysis(
                task_type=TaskType.GENERAL,
                complexity=0.2,
                required_capabilities=["general"],
                reasoning="Test routing decision.",
            ),
            reasoning="Test routing decision.",
        )


def test_evaluation_runs_every_baseline_and_routed_case() -> None:
    registry = ModelRegistry(
        models=[EvaluationModel("fast"), EvaluationModel("general"), EvaluationModel("reasoning")]
    )
    services = ApplicationServices(
        registry=registry,
        router=EvaluationRouter(),  # type: ignore[arg-type]
        executor=ModelExecutor(registry=registry, max_retries=0),
    )

    report = asyncio.run(run_evaluation(services))
    records = report["records"]
    summary = report["summary"]

    assert len(records) == 12
    assert sum(record["strategy"] == "fixed_model" for record in records) == 9
    assert sum(record["strategy"] == "routing" for record in records) == 3
    assert all(record["succeeded"] for record in records)
    assert summary["successful_runs"] == 12
    assert summary["routing_correct_tier_count"] == 3
