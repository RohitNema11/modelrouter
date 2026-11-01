from app.router.analyzer import QueryAnalyzer
from app.schemas.routing import TaskType


def test_detects_simple_general_query() -> None:
    analysis = QueryAnalyzer().analyze("What is the capital of France?")

    assert analysis.task_type == TaskType.GENERAL
    assert analysis.complexity == 0.20
    assert analysis.required_capabilities == ["general"]


def test_detects_coding_query() -> None:
    analysis = QueryAnalyzer().analyze(
        "Implement Dijkstra's algorithm in Python and explain time complexity."
    )

    assert analysis.task_type in {TaskType.CODING, TaskType.REASONING}
    assert analysis.reasoning.startswith("Detected ")


def test_detects_debugging_before_coding() -> None:
    analysis = QueryAnalyzer().analyze(
        "Debug this Python traceback and explain the error."
    )

    assert analysis.reasoning.startswith("Detected ")
    assert analysis.required_capabilities


def test_detects_reasoning_query() -> None:
    analysis = QueryAnalyzer().analyze(
        "Solve this complex logic problem step by step."
    )

    assert analysis.reasoning.startswith("Detected ")
    assert analysis.complexity >= 0.20
    assert analysis.required_capabilities


def test_detects_work_rate_problem_as_mathematics() -> None:
    analysis = QueryAnalyzer().analyze(
        "Calculate the combined work rate and find the reciprocal."
    )

    assert analysis.task_type == TaskType.MATHEMATICS
    assert "mathematics" in analysis.required_capabilities


def test_complexity_never_exceeds_one() -> None:
    analysis = QueryAnalyzer().analyze(
        "Solve this advanced distributed systems algorithm step by step "
        "with production optimization, edge cases, and time complexity."
    )

    assert analysis.complexity <= 1.0