from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.schemas.routing import RoutingDecision
from app.services import ApplicationServices, get_services

router = APIRouter(prefix="/api", tags=["routing"])


class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)


@router.post("/route", response_model=RoutingDecision)
async def route_query(
    request: RouteRequest,
    services: ApplicationServices = Depends(get_services),
) -> RoutingDecision:
    """Analyze a query and return a model selection without generating text."""
    return await services.router.route(request.query)
