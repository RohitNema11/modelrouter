from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.registry import create_model_registry
from app.router.analyzer import QueryAnalyzer
from app.router.router import Router
from app.schemas.routing import RoutingDecision

router = APIRouter(prefix="/api", tags=["routing"])


class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)


@router.post("/route", response_model=RoutingDecision)
async def route_query(request: RouteRequest) -> RoutingDecision:
    """Analyze a query and return a model selection without generating text."""
    try:
        routing_service = Router(
            registry=create_model_registry(),
            analyzer=QueryAnalyzer(),
        )
        return await routing_service.route(request.query)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error