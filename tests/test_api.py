"""
Integration tests for FastAPI scoring endpoints.
"""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "FinFraud Scoring API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data


def test_predict_endpoint_fraud():
    payload = {
        "step": 200,
        "type": "TRANSFER",
        "amount": 500000.0,
        "nameOrig": "C1928371",
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C9981273",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert data["fraud_probability"] > 0.5
    assert data["decision"] in ["DECLINE", "MANUAL_REVIEW"]
    assert len(data["top_factors"]) > 0


def test_predict_endpoint_legit():
    payload = {
        "step": 50,
        "type": "PAYMENT",
        "amount": 42.50,
        "nameOrig": "C1029384",
        "oldbalanceOrg": 1200.0,
        "newbalanceOrig": 1157.50,
        "nameDest": "M9281726",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "APPROVE"
    assert data["risk_tier"] == "LOW"
