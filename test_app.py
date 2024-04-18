import pytest
from app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_start_training(client):
    response = client.post("/start")
    assert response.status_code == 200
    assert b"Training started." in response.data

def test_stop_training(client):
    response = client.post("/stop")
    assert response.status_code == 200
    assert b"Training stopped and model reset." in response.data
