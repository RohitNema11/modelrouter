"""Application-scoped services and readiness checks."""

from dataclasses import dataclass
import logging

from fastapi import FastAPI, HTTPException, Request, status

from app.core.config import get_settings
from app.models.registry import ModelRegistry, create_model_registry
from app.reliability.executor import ModelExecutor
from app.router.analyzer import QueryAnalyzer
from app.router.router import Router

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplicationServices:
    """Long-lived dependencies shared by every API request."""

    registry: ModelRegistry
    router: Router
    executor: ModelExecutor


def create_application_services() -> ApplicationServices:
    """Build dependencies once during application startup."""
    settings = get_settings()
    registry = create_model_registry()
    analyzer = QueryAnalyzer()
    router = Router(
        registry=registry,
        analyzer=analyzer,
    )
    executor = ModelExecutor(
        registry=registry,
        max_retries=settings.max_retries,
        retry_backoff_seconds=settings.retry_backoff_seconds,
    )
    return ApplicationServices(
        registry=registry,
        router=router,
        executor=executor,
    )


def initialize_services(app: FastAPI) -> None:
    """Initialize services without preventing health diagnostics on failure."""
    try:
        app.state.services = create_application_services()
    except Exception as error:
        logger.exception("Application services could not be initialized.")
        app.state.services = None


def get_services(request: Request) -> ApplicationServices:
    """Return ready application services or a stable service-unavailable response."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "service_not_ready",
                "message": "The model-routing service is not ready.",
            },
            headers={"Retry-After": "5"},
        )
    return services
