"""
Unit tests for API
"""
import numpy as np
import pytest

from src.api import app

@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_ok(client):
    """GET /health should return 200 and status 'ok'."""
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.get_json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"


def test_predict_happy_path(client):
    """
    POST /predict with a valid (21, 3) points array should return 200
    and include 'letter' and 'confidence' in the response.
    """
    points = np.zeros((21, 3), dtype=float).tolist()

    resp = client.post("/predict", json={"points": points})
    assert resp.status_code == 200

    data = resp.get_json()
    assert isinstance(data, dict)
    assert "letter" in data
    assert "confidence" in data

    assert isinstance(data["letter"], str)
    assert isinstance(data["confidence"], (float, int))

    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_missing_body(client):
    """POST /predict with no JSON body should return 400."""
    resp = client.post("/predict")
    assert resp.status_code == 400

    data = resp.get_json()
    assert isinstance(data, dict)
    assert "error" in data
    assert "Invalid or missing JSON body" in data["error"]


def test_predict_missing_points_field(client):
    """POST /predict with JSON but no 'points' key should return 400."""
    resp = client.post("/predict", json={"not_points": []})
    assert resp.status_code == 400

    data = resp.get_json()
    assert isinstance(data, dict)
    assert "error" in data
    assert "Missing 'points' field" in data["error"]


def test_predict_points_wrong_type(client):
    """'points' must be a list; otherwise 400."""
    resp = client.post("/predict", json={"points": "not-a-list"})
    assert resp.status_code == 400

    data = resp.get_json()
    assert "error" in data
    assert "list of length 21" in data["error"]


def test_predict_points_wrong_length(client):
    """'points' list must have length 21; otherwise 400."""
    bad_points = [[0.0, 0.0, 0.0]] * 5

    resp = client.post("/predict", json={"points": bad_points})
    assert resp.status_code == 400

    data = resp.get_json()
    assert "error" in data
    assert "length 21" in data["error"]


def test_predict_points_wrong_shape(client):
    """
    'points' must be shape (21, 3). If inner lists are wrong length
    (e.g., 2 coords instead of 3), should return 400.
    """
    bad_points = [[0.0, 0.0] for _ in range(21)]

    resp = client.post("/predict", json={"points": bad_points})
    assert resp.status_code == 400

    data = resp.get_json()
    assert "error" in data
    assert "Expected 'points' shape (21, 3)" in data["error"]


def test_predict_points_non_numeric(client):
    """Non-numeric entries in 'points' should cause a 400 conversion error."""
    bad_points = [["a", "b", "c"] for _ in range(21)]

    resp = client.post("/predict", json={"points": bad_points})
    assert resp.status_code == 400

    data = resp.get_json()
    assert "error" in data
    assert "Could not convert 'points' to a float32 array" in data["error"]
