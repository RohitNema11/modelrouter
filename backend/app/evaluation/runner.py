"""Run a fixed baseline-versus-routing evaluation from the command line."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Literal

from app.reliability.executor import UpstreamGenerationError
from app.services import ApplicationServices, create_application_services
from app.telemetry.cost import estimate_cost


MODEL_TIERS = ("fast", "general", "reasoning")


@dataclass(frozen=True)
class EvaluationCase:
    """One prompt representing the expected routing tier."""

    name: str
    expected_tier: str
    query: str


DEFAULT_CASES = (
    EvaluationCase(
        name="fast_fact",
        expected_tier="fast",
        query="What is the capital of France?",
    ),
    EvaluationCase(
        name="general_coding",
        expected_tier="general",
        query="Write a Python function to reverse a string.",
    ),
    EvaluationCase(
        name="reasoning_algorithm",
        expected_tier="reasoning",
        query=(
            "Implement Dijkstra's algorithm in Python and explain its time "
            "complexity."
        ),
    ),
)


@dataclass(frozen=True)
class EvaluationRecord:
    """One model invocation captured for comparison and later display."""

    case_name: str
    expected_tier: str
    strategy: Literal["fixed_model", "routing"]
    requested_model: str | None
    selected_model: str | None
    response: str | None
    succeeded: bool
    error_reason: str | None
    attempts: int
    fallback_used: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    latency_ms: float


async def run_evaluation(
    services: ApplicationServices,
    cases: tuple[EvaluationCase, ...] = DEFAULT_CASES,
) -> dict[str, object]:
    """Run fixed-model baselines first, then routing for every evaluation case."""
    records: list[EvaluationRecord] = []

    for case in cases:
        for tier in MODEL_TIERS:
            records.append(await _run_fixed_model(services, case, tier))

    for case in cases:
        records.append(await _run_routing(services, case))

    return {
        "cases": [asdict(case) for case in cases],
        "records": [asdict(record) for record in records],
        "summary": _summarize(records),
    }


async def _run_fixed_model(
    services: ApplicationServices,
    case: EvaluationCase,
    tier: str,
) -> EvaluationRecord:
    """Run a baseline against one exact tier without changing the selected tier."""
    return await _execute(
        services=services,
        case=case,
        strategy="fixed_model",
        selected_model=tier,
        required_capabilities=[],
        requested_model=tier,
        allow_fallback=False,
    )


async def _run_routing(
    services: ApplicationServices,
    case: EvaluationCase,
) -> EvaluationRecord:
    """Route and run one case using the normal application policy."""
    decision = await services.router.route(case.query)
    return await _execute(
        services=services,
        case=case,
        strategy="routing",
        selected_model=decision.selected_model,
        required_capabilities=decision.analysis.required_capabilities,
        requested_model=None,
        allow_fallback=True,
    )


async def _execute(
    services: ApplicationServices,
    case: EvaluationCase,
    strategy: Literal["fixed_model", "routing"],
    selected_model: str,
    required_capabilities: list[str],
    requested_model: str | None,
    allow_fallback: bool,
) -> EvaluationRecord:
    started_at = time.perf_counter()
    try:
        execution = await services.executor.generate(
            selected_model_name=selected_model,
            prompt=case.query,
            required_capabilities=required_capabilities,
            allow_fallback=allow_fallback,
        )
    except UpstreamGenerationError as error:
        return _failed_record(
            case=case,
            strategy=strategy,
            requested_model=requested_model,
            selected_model=selected_model,
            error_reason=error.reason,
            attempts=error.attempts,
            latency_ms=_elapsed_ms(started_at),
        )
    except Exception:
        return _failed_record(
            case=case,
            strategy=strategy,
            requested_model=requested_model,
            selected_model=selected_model,
            error_reason="evaluation_error",
            attempts=0,
            latency_ms=_elapsed_ms(started_at),
        )

    response = execution.response
    metadata = services.registry.get(response.model_name).metadata
    cost = estimate_cost(
        metadata=metadata,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
    return EvaluationRecord(
        case_name=case.name,
        expected_tier=case.expected_tier,
        strategy=strategy,
        requested_model=requested_model,
        selected_model=response.model_name,
        response=response.content,
        succeeded=True,
        error_reason=None,
        attempts=execution.attempts,
        fallback_used=execution.fallback_used,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        estimated_cost_usd=round(cost, 8) if cost is not None else None,
        latency_ms=_elapsed_ms(started_at),
    )


def _failed_record(
    case: EvaluationCase,
    strategy: Literal["fixed_model", "routing"],
    requested_model: str | None,
    selected_model: str,
    error_reason: str,
    attempts: int,
    latency_ms: float,
) -> EvaluationRecord:
    return EvaluationRecord(
        case_name=case.name,
        expected_tier=case.expected_tier,
        strategy=strategy,
        requested_model=requested_model,
        selected_model=selected_model,
        response=None,
        succeeded=False,
        error_reason=error_reason,
        attempts=attempts,
        fallback_used=False,
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=None,
        latency_ms=latency_ms,
    )


def _summarize(records: list[EvaluationRecord]) -> dict[str, object]:
    """Return compact aggregate metrics for the evaluation display."""
    successful_records = [record for record in records if record.succeeded]
    routed_records = [record for record in records if record.strategy == "routing"]
    costs = [record.estimated_cost_usd for record in successful_records]
    known_costs = [cost for cost in costs if cost is not None]

    return {
        "total_runs": len(records),
        "successful_runs": len(successful_records),
        "failed_runs": len(records) - len(successful_records),
        "average_latency_ms": round(
            sum(record.latency_ms for record in records) / len(records), 2
        ) if records else 0,
        "total_estimated_cost_usd": round(sum(known_costs), 8),
        "routing_correct_tier_count": sum(
            record.succeeded and record.selected_model == record.expected_tier
            for record in routed_records
        ),
        "routing_model_selection": dict(
            sorted(Counter(
                record.selected_model for record in routed_records if record.selected_model
            ).items())
        ),
    }


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1_000, 2)


def main() -> None:
    """Run the standard evaluation and print or persist its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report; stdout is used when omitted.",
    )
    args = parser.parse_args()

    report = asyncio.run(run_evaluation(create_application_services()))
    rendered_report = json.dumps(report, indent=2)
    if args.output is None:
        print(rendered_report)
    else:
        args.output.write_text(rendered_report + "\n", encoding="utf-8")
        print(f"Evaluation report written to {args.output}")


if __name__ == "__main__":
    main()
