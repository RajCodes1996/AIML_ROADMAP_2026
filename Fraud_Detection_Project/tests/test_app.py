
# ==================================================
# DAY 24 - AUTOMATED API TESTING
# ==================================================

from fastapi.testclient import TestClient
from src.app import app, features

client = TestClient(app)


# Test 1: Check whether API is running
def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test 2: Check whether prediction works
def test_prediction():

    # Create sample input with every required feature
    sample = {
        feature: 0.0 for feature in features
    }

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "threshold" in result
    assert "prediction" in result

    assert result["prediction"] in [
        "Fraud",
        "Legitimate"
    ]


# Test 3: Check missing feature validation
def test_missing_feature():

    sample = {
        feature: 0.0 for feature in features
    }

    # Remove one required feature
    sample.pop(features[0])

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 422


# Test 4: Check unexpected feature validation
def test_extra_feature():

    sample = {
        feature: 0.0 for feature in features
    }

    sample["unexpected_feature"] = 123

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 422