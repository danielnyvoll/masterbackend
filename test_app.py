import pytest
from app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_hello_world(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hello, World!" in response.data
