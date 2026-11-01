from app.core.config import get_settings
from app.models.base import BaseModel


class ModelRegistry:
    """Central catalogue of models available to ModelRouter."""

    def __init__(self, models: list[BaseModel]) -> None:
        self._models = {model.metadata.name: model for model in models}

    def get(self, model_name: str) -> BaseModel:
        """Return a model by name."""
        try:
            return self._models[model_name]
        except KeyError as error:
            raise ValueError(f"Unknown model: {model_name}") from error

    def list_models(self) -> list[BaseModel]:
        """Return every registered model."""
        return list(self._models.values())


def create_model_registry() -> ModelRegistry:
    """Create a registry backed by Gemini."""
    settings = get_settings()

    if settings.gemini_api_key is None:
        raise ValueError(
            "GEMINI_API_KEY must be configured before using the application."
        )

    # Keep the provider SDK out of import-time paths such as /health and tooling.
    from app.models.gemini import create_gemini_models

    models = create_gemini_models(
        api_key=settings.gemini_api_key.get_secret_value(),
        timeout_seconds=settings.request_timeout_seconds,
        reasoning_timeout_seconds=settings.reasoning_timeout_seconds,
    )
    return ModelRegistry(models=models)
