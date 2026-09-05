import time

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.registry import create_model_registry
from app.reliability.executor import ModelExecutor
from app.router.analyzer import QueryAnalyzer
from app.router.router import Router
from app.schemas.chat import ChatRequest, ChatResponse, TokenUsage
from app.telemetry.cost import estimate_cost

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Route a query, generate a response, and return execution metadata."""
    total_start = time.perf_counter()
    settings = get_settings()

    registry = create_model_registry()
    routing_service = Router(
        registry=registry,
        analyzer=QueryAnalyzer(),
    )

    try:
        routing_start = time.perf_counter()
        routing_decision = await routing_service.route(request.query)
        routing_latency_ms = (time.perf_counter() - routing_start) * 1_000

        model_start = time.perf_counter()
        executor = ModelExecutor(
            registry=registry,
            max_retries=settings.max_retries,
        )
        execution = await executor.generate(
            selected_model_name=routing_decision.selected_model,
            prompt=request.query,
            required_capabilities=routing_decision.analysis.required_capabilities,
        )
        model_latency_ms = (time.perf_counter() - model_start) * 1_000

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    total_latency_ms = (time.perf_counter() - total_start) * 1_000
    model_response = execution.response
    executed_model = registry.get(model_response.model_name)

    estimated_cost_usd = estimate_cost(
        metadata=executed_model.metadata,
        input_tokens=model_response.input_tokens,
        output_tokens=model_response.output_tokens,
    )

    return ChatResponse(
        response=model_response.content,
        selected_model=model_response.model_name,
        initial_model=execution.initial_model,
        fallback_used=execution.fallback_used,
        attempts=execution.attempts,
        routing=routing_decision,
        tokens=TokenUsage(
            input_tokens=model_response.input_tokens,
            output_tokens=model_response.output_tokens,
            total_tokens=(
                model_response.input_tokens + model_response.output_tokens
            ),
        ),
        estimated_cost_usd=round(estimated_cost_usd, 8),
        routing_latency_ms=round(routing_latency_ms, 2),
        model_latency_ms=round(model_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )