import time

from fastapi import APIRouter, HTTPException

from app.models.registry import create_model_registry
from app.router.analyzer import QueryAnalyzer
from app.router.router import Router
from app.schemas.chat import ChatRequest, ChatResponse, TokenUsage

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Route a query, generate a response, and return execution metadata."""
    total_start = time.perf_counter()

    registry = create_model_registry()
    routing_service = Router(
        registry=registry,
        analyzer=QueryAnalyzer(),
    )

    try:
        routing_start = time.perf_counter()
        routing_decision = routing_service.route(request.query)
        routing_latency_ms = (time.perf_counter() - routing_start) * 1_000

        selected_model = registry.get(routing_decision.selected_model)

        model_start = time.perf_counter()
        model_response = await selected_model.generate(request.query)
        model_latency_ms = (time.perf_counter() - model_start) * 1_000

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    total_latency_ms = (time.perf_counter() - total_start) * 1_000

    input_tokens = model_response.input_tokens
    output_tokens = model_response.output_tokens

    return ChatResponse(
        response=model_response.content,
        selected_model=model_response.model_name,
        routing=routing_decision,
        tokens=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        routing_latency_ms=round(routing_latency_ms, 2),
        model_latency_ms=round(model_latency_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
    )