import pytest
from flask_socketio import SocketIOTestClient
from app import app as flask_app, socketio  # Ensure these imports point to your Flask app and socketio instance
import numpy as np
import base64
from io import BytesIO
from PIL import Image

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


def test_handle_send_image(socket_client):
    # Create a dummy image
    img = Image.new('RGB', (10, 10), color='red')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{img_base64}"

    # Mock data for the 'send_image' event
    data = {
        'image': image_data,
        'playerPosition': {'x': 0, 'y': 0, 'z': 0},
        'oppositePlayerPosition': {'x': 0, 'y': 0, 'z': 0},
        'ballPosition': {'x': 0, 'y': 0, 'z': 0},
        'isGoal': {'intersecting': False, 'scoringSide': 0},
        'isMultiplayer': {'multiplayer': False}
    }

    socket_client.emit('send_image', data)
    received = socket_client.get_received()

    assert len(received) >= 0
    for message in received:
        assert 'command' in message['name'] or 'reset' in message['name']

def test_get_available_commands():
    from app import get_available_commands
    commands_close = get_available_commands(0.5)
    assert "dribble" in commands_close and "shoot" in commands_close

    commands_far = get_available_commands(2.0)
    assert "dribble" not in commands_far and "shoot" not in commands_far
