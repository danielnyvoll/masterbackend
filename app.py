from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Q-table with zeros for simplicity
q_table = np.zeros((24, 14, 24, 14, 4))

def grid_map(value, min_value, max_value, grid_size):
    """Map a value from real coordinates to a grid index."""
    # Scale the value from [min_value, max_value] to [0, grid_size - 1]
    scale = (value - min_value) / (max_value - min_value)
    return int(scale * grid_size)

def determine_next_action(player_pos, ball_pos):
    # Define the grid size for simplification
    grid_size_x = 24  # For the -120 to 120 range
    grid_size_y = 14  # For the -70 to 70 range
    
    # Map real-world positions to grid indices
    player_state_x = grid_map(player_pos['x'], -120, 120, grid_size_x)
    player_state_y = grid_map(player_pos['y'], -70, 70, grid_size_y)
    ball_state_x = grid_map(ball_pos['x'], -120, 120, grid_size_x)
    ball_state_y = grid_map(ball_pos['y'], -70, 70, grid_size_y)
    
    # Determine the action with the highest Q-value for the current state
    action_index = np.argmax(q_table[player_state_x, player_state_y, ball_state_x, ball_state_y])
    
    # Map the action index to a command
    actions = ["up", "down", "left", "right"]
    return actions[action_index]

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('update_positions')
def handle_update_positions(json):
    # Assuming json is a dictionary with 'playerPosition' and 'ballPosition' keys
    if json is not None:
        player_position = json.get('playerPosition')
        ball_position = json.get('ballPosition')
        if player_position and ball_position:
            action = determine_next_action(player_position, ball_position)
            emit('command', action)  # Send the determined action back to the client

if __name__ == '__main__':
    socketio.run(app)
