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
    assert decision.analysis.task_type == TaskType.CODING
    assert decision.analysis.complexity > 0.0
    assert "coding" in decision.analysis.required_capabilities
    assert "keyword rules" in decision.analysis.reasoning


def test_routes_complex_algorithm_query_to_reasoning_model() -> None:
    decision = asyncio.run(
        create_router().route(
            "Implement Dijkstra's algorithm in Python and explain time complexity."
        )
    )

    assert decision.selected_model == "reasoning"
    assert decision.analysis.task_type == TaskType.CODING
    assert decision.analysis.complexity > 0.0
    assert "coding" in decision.analysis.required_capabilities
    assert 0.75 <= decision.confidence <= 1.0


def test_routes_debugging_query_to_general_model() -> None:
    decision = asyncio.run(
        create_router().route(
            "Debug this Python traceback and explain the error."
        )
    )

    assert decision.selected_model in {"fast", "general", "reasoning"}
    assert decision.analysis.task_type == TaskType.DEBUGGING
    assert decision.analysis.complexity > 0.0
    assert "debugging" in decision.analysis.required_capabilities


def test_routing_explanation_mentions_selected_model() -> None:
    decision = asyncio.run(create_router().route("Summarize this article."))

    assert decision.selected_model in decision.reasoning
    assert (
        "classifier decision" in decision.reasoning
        or "rule-based complexity fallback" in decision.reasoning
    )


def test_low_classifier_confidence_uses_complexity_fallback() -> None:
    analyzer = QueryAnalyzer(use_classifier=False)
    router = Router(create_model_registry(), analyzer)
    decision = asyncio.run(router.route("An ambiguous query."))

    assert decision.selected_model == "fast"
    assert decision.analysis.task_type == TaskType.GENERAL
    assert "rule-based complexity fallback" in decision.reasoning


def test_classifier_confidence_above_threshold_uses_classifier() -> None:
    analyzer = QueryAnalyzer()
    analyzer.classify = MethodType(
        lambda self, query: ("reasoning", 0.73),
        analyzer,
    )
    router = Router(create_model_registry(), analyzer)

    decision = asyncio.run(router.route("Explain the Josephus problem step by step."))

    assert decision.selected_model == "reasoning"
    assert decision.confidence == 0.73
    assert "classifier decision" in decision.reasoning


def test_classifier_failure_uses_complexity_rule_fallback() -> None:
    analyzer = QueryAnalyzer(use_classifier=False)
    router = Router(create_model_registry(), analyzer)
    decision = asyncio.run(router.route("Write a Python function."))

    assert decision.selected_model == "general"
    assert "rule-based complexity fallback" in decision.reasoning