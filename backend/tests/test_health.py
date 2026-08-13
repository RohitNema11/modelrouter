from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
    }

def test_list_models() -> None:
    response = client.get("/api/models")

    assert response.status_code == 200

    models = response.json()

    assert [model["name"] for model in models] == [
        "fast",
        "general",
        "reasoning",
    ]
    assert models[0]["provider"] == "mock"
    assert models[2]["context_window"] == 131_072