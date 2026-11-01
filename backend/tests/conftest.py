import pytest

from app.core.config import get_settings
from app.models.base import ModelResponse
from app.models.gemini import GeminiModel


@pytest.fixture(autouse=True)
def use_test_gemini_client(monkeypatch: pytest.MonkeyPatch):
    """Keep tests deterministic without making requests to Gemini."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    class FakeGeminiClient:
        """Avoid constructing the real SDK transport in unit tests."""

        def __init__(self, *args, **kwargs) -> None:
            self.aio = None

    monkeypatch.setattr("app.models.gemini.genai.Client", FakeGeminiClient)

    async def generate(self: GeminiModel, prompt: str) -> ModelResponse:
        if prompt.startswith("Choose a tier:"):
            query = prompt.split("Query: ", 1)[1].lower()
            if any(word in query for word in ("dijkstra", "algorithm", "complexity")):
                content = "reasoning"
            elif any(word in query for word in ("python", "code", "debug", "function")):
                content = "general"
            else:
                content = "fast"
        else:
            content = f"Test response for: {prompt}"

        return ModelResponse(
            content=content,
            input_tokens=len(prompt.split()),
            output_tokens=len(content.split()),
            model_name=self.metadata.name,
        )

    monkeypatch.setattr(GeminiModel, "generate", generate)

    yield

    get_settings.cache_clear()
