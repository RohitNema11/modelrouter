from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status

from app.core.config import get_settings

from app.api.models import router as models_router

from app.api.routing import router as routing_router

from app.api.chat import router as chat_router
from app.services import initialize_services

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare shared dependencies before the application accepts requests."""
    initialize_services(app)
    yield

app = FastAPI(
    title=settings.app_name,
    description="API for routing requests across LLM model tiers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(models_router)
app.include_router(routing_router)
app.include_router(chat_router)

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {
        "status": "ok",
        "environment": settings.environment,
    }


@app.get("/ready")
async def readiness_check(response: Response) -> dict[str, str]:
    """Report whether model-routing dependencies initialized successfully."""
    if getattr(app.state, "services", None) is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "environment": settings.environment,
        }

    return {
        "status": "ready",
        "environment": settings.environment,
    }
