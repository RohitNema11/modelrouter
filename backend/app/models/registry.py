from app.models.base import BaseModel
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
    """Create the default registry for the local development environment."""
    return ModelRegistry(
        models=[
            create_fast_model(),
            create_general_model(),
            create_reasoning_model(),
        ]
    )