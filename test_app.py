import pytest
from flask_socketio import SocketIOTestClient
from app import app as flask_app, socketio  # Make sure these imports correctly point to your Flask app and socketio instance
import numpy as np

@pytest.fixture(scope='module')
def client():
    """Create and configure a test client for the Flask application."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

@pytest.fixture(scope='module')
def socket_client():
    """Fixture to create a SocketIO test client."""
    client = SocketIOTestClient(flask_app, socketio)
    yield client
    client.disconnect()

def test_start_training(client):
    response = client.post("/start", json={"model": "q-learning"})
    assert response.status_code == 200
    assert "Training started." in response.json['message']

def test_stop_training(client):
    response = client.post("/stop")
    assert response.status_code == 200
    assert "Training stopped and model reset." in response.json['message']

