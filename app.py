from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import os

# Import your Deep Q-Learning Model
from models.deep_q_learning import DeepQLearningModel

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Initialize your model
current_model = DeepQLearningModel(state_size=4, action_size=4, learning_rate=0.001)
batch_size = 32  # Define your batch size for training

# Placeholder variables for distance calculations
previous_ball_distance_to_goal = float('inf')
previous_player_distance_to_ball = float('inf')

# Command mapping from model output to action commands
command_mapping = {0: "up", 1: "left", 2: "down", 3: "right"}

def calculate_distance(pos1, pos2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((pos2['x'] - pos1['x'])**2 + (pos2['y'] - pos1['y'])**2)

def get_reward(player_pos, ball_pos, isGoal, prev_ball_dist_to_goal, prev_player_dist_to_ball):
    """Define your reward function here."""
    # Example reward calculation
    reward = 100 if isGoal else -1
    return reward, calculate_distance(ball_pos, {'x': 0, 'y': 0}), calculate_distance(player_pos, ball_pos)

@app.route('/save', methods=['POST'])
def save_model():
    """Save the current model's weights."""
    current_model.save('model_weights.h5')
    return jsonify({"message": "Model weights saved."})

@app.route('/load', methods=['GET'])
def load_model():
    """Load model weights."""
    if os.path.exists('model_weights.h5'):
        current_model.load('model_weights.h5')
        return jsonify({"message": "Model weights loaded."})
    return jsonify({"error": "Model weights file not found."})
# Global storage for current and next states
current_state = None
next_state = None
last_action = None
last_reward = None

@socketio.on('update_positions')
def handle_update_positions(data):
    global current_model, last_action, last_reward, current_state, next_state

    player_position = data.get('playerPosition')
    ball_position = data.get('ballPosition')
    isGoal = data.get('isGoal', {}).get('intersecting', False)

    # If this is the first message or after a reset, initialize the current state
    if current_state is None:
        current_state = np.array([player_position['x'], player_position['y'], ball_position['x'], ball_position['y']])
        current_state = np.reshape(current_state, [1, 4])
        return  # Wait for the next message which reflects the result of an action

    # If we have an action waiting for its result
    if last_action is not None:
        next_state = np.array([player_position['x'], player_position['y'], ball_position['x'], ball_position['y']])
        next_state = np.reshape(next_state, [1, 4])

        # Update model with the transition
        current_model.remember(current_state, last_action, last_reward, next_state, isGoal)
        
        if len(current_model.memory) > batch_size:
            current_model.replay(batch_size)
        
        # Prepare for the next action
        current_state = next_state
        last_action = None
        last_reward = None

    # Calculate reward based on the current action and state
    reward, _, _ = get_reward(player_position, ball_position, isGoal, previous_ball_distance_to_goal, previous_player_distance_to_ball)

    # Choose and emit the next action
    action = current_model.choose_action(current_state)
    command = command_mapping.get(action, "unknown")
    emit('command', command)

    # Temporarily store last action and reward
    last_action = action
    last_reward = reward


if __name__ == '__main__':
    socketio.run(app, debug=True)
