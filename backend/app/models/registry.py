from app.core.config import get_settings
from app.models.base import BaseModel
from app.models.groq import create_groq_models
from app.models.mock import (
    create_fast_model,
    create_general_model,
    create_reasoning_model,
)


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
    """Create a registry using the configured model provider."""
    settings = get_settings()

    if settings.model_provider == "mock":
        models = [
            create_fast_model(),
            create_general_model(),
            create_reasoning_model(),
        ]
    elif settings.model_provider == "groq":
        if settings.groq_api_key is None:
            raise ValueError(
                "GROQ_API_KEY must be configured when MODEL_PROVIDER=groq."
            )

        models = create_groq_models(
            api_key=settings.groq_api_key.get_secret_value(),
            timeout_seconds=settings.request_timeout_seconds,
        )
    else:
        raise ValueError(
            "MODEL_PROVIDER must be either 'mock' or 'groq'."
        )

    return ModelRegistry(models=models)