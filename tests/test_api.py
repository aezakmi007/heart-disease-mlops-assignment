import pytest
from fastapi.testclient import TestClient

import app as api

VALID = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


@pytest.fixture
def client(trained_model):
    api.state["model"] = trained_model
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range_value(client):
    assert client.post("/predict", json={**VALID, "age": 500}).status_code == 422


def test_predict_rejects_invalid_thal_code(client):
    assert client.post("/predict", json={**VALID, "thal": 5}).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=VALID)
    body = client.get("/metrics").text
    assert "api_requests_total" in body
    assert "api_predictions_total" in body
