from fastapi import APIRouter, Depends

from app.services import ApplicationServices, get_services

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models(
    services: ApplicationServices = Depends(get_services),
) -> list[dict[str, object]]:
    """Return metadata for every registered model."""
    return [
        {
            "name": model.metadata.name,
            "model_id": model.metadata.model_id,
            "provider": model.metadata.provider,
            "tier": model.metadata.tier,
            "context_window": model.metadata.context_window,
            "capabilities": sorted(model.metadata.capabilities),
            "reliability_score": model.metadata.reliability_score,
            "expected_latency_ms": model.metadata.expected_latency_ms,
        }
        for model in services.registry.list_models()
    ]
