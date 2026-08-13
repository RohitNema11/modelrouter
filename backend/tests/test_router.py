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
    decision = create_router().route("What is the capital of France?")

    assert decision.selected_model == "fast"
    assert decision.analysis.task_type == TaskType.GENERAL
    assert decision.confidence == 0.90


def test_routes_moderate_coding_query_to_general_model() -> None:
    decision = create_router().route("Write a Python function to reverse a string.")

    assert decision.selected_model == "general"
    assert decision.analysis.task_type == TaskType.CODING
    assert decision.analysis.complexity == 0.45


def test_routes_complex_algorithm_query_to_reasoning_model() -> None:
    decision = create_router().route(
        "Implement Dijkstra's algorithm in Python and explain time complexity."
    )

    assert decision.selected_model == "reasoning"
    assert "complex_reasoning" in decision.analysis.required_capabilities
    assert decision.confidence == 0.90


def test_routes_debugging_query_to_general_model() -> None:
    decision = create_router().route(
        "Debug this Python traceback and explain the error."
    )

    assert decision.selected_model == "general"
    assert decision.analysis.task_type == TaskType.DEBUGGING


def test_routing_explanation_mentions_selected_model() -> None:
    decision = create_router().route("Summarize this article.")

    assert "fast" in decision.reasoning
    assert "summarization" in decision.reasoning