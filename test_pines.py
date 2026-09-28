from fastapi.testclient import TestClient
from pines import app, classification_threshold, current_model, model_name

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"].startswith("Welcome to the Pines NLP Model\n\n Current Model:")

def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"
    assert response.json()["model"] == model_name
    assert response.json()["classification_threshold"] == classification_threshold

def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "the patient has vte"})
        assert response.status_code == 200
        assert response.json()["model"] == model_name
        assert response.json()["classification_threshold"] == classification_threshold
        assert response.json()["prediction"]["label"] == 1

def test_list_models_returns_loaded_model():
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["current_model"] == current_model
    assert len(body["models"]) == 1
    assert body["models"][0]["id"] == current_model
    assert body["models"][0]["name"] == model_name
    assert body["models"][0]["classification_threshold"] == classification_threshold
    assert body["models"][0]["search_query"]

def test_model_search_query():
    response = client.get(f"/models/{current_model}/search_query")
    assert response.status_code == 200
    assert response.json()["model"] == current_model
    assert response.json()["search_query"].startswith("dvt OR")

def test_model_search_query_unknown_model():
    assert client.get("/models/NOT-A-MODEL/search_query").status_code == 404

def test_predict_batch():
    with TestClient(app) as client:
        response = client.post("/predict_batch", json=[{"text": "the patient has vte"},
                                                       {"text": "the patient has no PE"}])
        assert response.status_code == 200
        assert response.json()["model"] == model_name
        assert response.json()["classification_threshold"] == classification_threshold
        assert response.json()["prediction"][0]["label"] == 1
        assert response.json()["prediction"][1]["label"] == 0
