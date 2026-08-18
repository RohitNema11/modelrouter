from pydantic import BaseModel, Field

from app.schemas.routing import RoutingDecision


class ChatRequest(BaseModel):
    """A user's chat request."""

    query: str = Field(min_length=1, max_length=20_000)


class TokenUsage(BaseModel):
    """Normalized token usage for a generated response."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatResponse(BaseModel):
    """The answer plus the routing details that produced it."""

    response: str
    selected_model: str
    initial_model: str
    fallback_used: bool
    attempts: int = Field(ge=1)
    routing: RoutingDecision
    tokens: TokenUsage
    estimated_cost_usd: float = Field(ge=0)
    routing_latency_ms: float = Field(ge=0)
    model_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    