from fastapi import FastAPI

from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API for routing requests across LLM model tiers.",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {
        "status": "ok",
        "environment": settings.environment,
    }