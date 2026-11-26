"""FastAPI inference endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.config import MODEL_PIPELINE_PATH

VALID_PAYLOAD = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


@pytest.mark.skipif(
    not MODEL_PIPELINE_PATH.exists(),
    reason="Trained model artifacts not generated yet; run train.py first.",
)
def test_api_health_and_predictions():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        h_json = health.json()
        assert h_json["status"] == "healthy"
        assert h_json["model_loaded"] is True

        info = client.get("/model-info")
        assert info.status_code == 200
        assert "champion_model" in info.json()

        pred = client.post("/predict", json=VALID_PAYLOAD)
        assert pred.status_code == 200
        p_json = pred.json()
        assert p_json["predicted_price_usd"] > 0.0
        assert p_json["interval_90_low_usd"] <= p_json["interval_90_high_usd"]
        assert len(p_json["top_drivers"]) == 5

        batch = client.post(
            "/predict/batch",
            json={"records": [VALID_PAYLOAD, VALID_PAYLOAD], "include_shap": False},
        )
        assert batch.status_code == 200
        b_json = batch.json()
        assert b_json["count"] == 2
        assert len(b_json["predictions"]) == 2


def test_api_rejects_invalid_california_coordinates():
    invalid_payload = dict(VALID_PAYLOAD)
    invalid_payload["Latitude"] = 10.0  # Outside California [32.0, 42.5]
    with TestClient(app) as client:
        resp = client.post("/predict", json=invalid_payload)
        assert resp.status_code == 422
