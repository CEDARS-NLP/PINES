from fastapi.testclient import TestClient
from pines import app, model_name

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("Welcome to the Pines NLP Model\n\n Current Model:")

def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "the patient has vte"})
        assert response.status_code == 200
        assert response.json()["model"] == model_name
        assert response.json()["prediction"]["label"] == 1

def test_predict_batch():
    with TestClient(app) as client:
        response = client.post("/predict_batch", json=[{"text": "the patient has vte"},
                                                       {"text": "the patient has no PE"}])
        assert response.status_code == 200
        assert response.json()["prediction"][0]["label"] == 1
        assert response.json()["prediction"][1]["label"] == 0
