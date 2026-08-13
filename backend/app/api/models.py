from fastapi import APIRouter

from app.models.registry import create_model_registry

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models() -> list[dict[str, object]]:
    """Return metadata for every registered model."""
    registry = create_model_registry()

    return [
        {
            "name": model.metadata.name,
            "provider": model.metadata.provider,
            "tier": model.metadata.tier,
            "context_window": model.metadata.context_window,
            "capabilities": sorted(model.metadata.capabilities),
            "reliability_score": model.metadata.reliability_score,
            "expected_latency_ms": model.metadata.expected_latency_ms,
        }
        for model in registry.list_models()
    ]