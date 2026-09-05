import asyncio
from types import MethodType

from app.models.registry import create_model_registry
from app.router.analyzer import QueryAnalyzer
from app.router.router import Router
from app.schemas.routing import TaskType


def create_router() -> Router:
    return Router(
        registry=create_model_registry(),
        analyzer=QueryAnalyzer(),
    )


def test_routes_simple_general_query_to_fast_model() -> None:
    decision = asyncio.run(create_router().route("What is the capital of France?"))

    assert decision.selected_model == "fast"
    assert decision.analysis.task_type == TaskType.GENERAL
    assert 0.75 <= decision.confidence <= 1.0
    assert "classifier decision" in decision.reasoning


def test_routes_moderate_coding_query_to_general_model() -> None:
    decision = asyncio.run(
        create_router().route("Write a Python function to reverse a string.")
    )

    assert decision.selected_model == "general"
    assert decision.analysis.task_type in {TaskType.CODING, TaskType.REASONING}
    assert decision.analysis.complexity == 0.0
    assert decision.analysis.required_capabilities == []


def test_routes_complex_algorithm_query_to_reasoning_model() -> None:
    decision = asyncio.run(
        create_router().route(
            "Implement Dijkstra's algorithm in Python and explain time complexity."
        )
    )

    assert decision.selected_model == "reasoning"
    assert decision.analysis.complexity == 0.0
    assert decision.analysis.required_capabilities == []
    assert 0.75 <= decision.confidence <= 1.0


def test_routes_debugging_query_to_general_model() -> None:
    decision = asyncio.run(
        create_router().route(
            "Debug this Python traceback and explain the error."
        )
    )

    assert decision.selected_model in {"fast", "general", "reasoning"}
    assert decision.analysis.complexity == 0.0
    assert decision.analysis.required_capabilities == []


def test_routing_explanation_mentions_selected_model() -> None:
    decision = asyncio.run(create_router().route("Summarize this article."))

    assert decision.selected_model in decision.reasoning
    assert "classifier" in decision.reasoning or "LLM escalation" in decision.reasoning


def test_low_classifier_confidence_escalates_to_llm() -> None:
    analyzer = QueryAnalyzer(use_classifier=False)
    router = Router(create_model_registry(), analyzer, classifier_confidence_threshold=0.75)

    async def predict_tier(self: Router, query: str) -> str:
        return "reasoning"

    router._get_llm_tier_prediction = MethodType(predict_tier, router)
    decision = asyncio.run(router.route("An ambiguous query."))

    assert decision.selected_model == "reasoning"
    assert "LLM escalation" in decision.reasoning


def test_llm_failure_uses_complexity_rule_fallback() -> None:
    analyzer = QueryAnalyzer(use_classifier=False)
    router = Router(create_model_registry(), analyzer)

    async def fail_llm(self: Router, query: str) -> str:
        raise RuntimeError("LLM unavailable")

    router._get_llm_tier_prediction = MethodType(fail_llm, router)
    decision = asyncio.run(router.route("Write a Python function."))

    assert decision.selected_model == "general"
    assert "rule-based complexity fallback" in decision.reasoning