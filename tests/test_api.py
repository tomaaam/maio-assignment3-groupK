from fastapi.testclient import TestClient

from src.app import app


VALID_INPUT = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def test_valid_request():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_missing_field_returns_422():
    input_data = VALID_INPUT.copy()
    del input_data["s6"]

    with TestClient(app) as client:
        response = client.post("/predict", json=input_data)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "v0.1",
    }