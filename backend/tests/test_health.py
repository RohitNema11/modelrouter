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

def test_route_query() -> None:
    response = client.post(
        "/api/route",
        json={
            "query": (
                "Implement Dijkstra's algorithm in Python and "
                "explain time complexity."
            )
        },
    )

    assert response.status_code == 200

    decision = response.json()

    assert decision["selected_model"] == "reasoning"
    assert decision["analysis"]["task_type"] == "coding"
    assert decision["analysis"]["complexity"] == 0.65
    assert decision["confidence"] == 0.90


def test_route_query_rejects_empty_query() -> None:
    response = client.post("/api/route", json={"query": ""})

    assert response.status_code == 422

def test_chat_query_returns_response_and_metadata() -> None:
    response = client.post(
        "/api/chat",
        json={"query": "Write a Python function to reverse a string."},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["selected_model"] == "general"
    assert result["response"].startswith("[Mock general response]")
    assert result["routing"]["analysis"]["task_type"] == "coding"

    assert result["tokens"]["input_tokens"] == 8
    assert result["tokens"]["output_tokens"] > 0
    assert result["tokens"]["total_tokens"] == (
        result["tokens"]["input_tokens"] + result["tokens"]["output_tokens"]
    )

    assert result["routing_latency_ms"] >= 0
    assert result["model_latency_ms"] >= 0
    assert result["total_latency_ms"] >= 0


def test_chat_query_rejects_empty_query() -> None:
    response = client.post("/api/chat", json={"query": ""})

    assert response.status_code == 422