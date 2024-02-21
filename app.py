from flask import Flask
from flask_socketio import SocketIO, emit
import numpy as np
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Q-table with zeros for simplicity
q_table = np.zeros((24, 14, 24, 14, 4))  # States: Player X, Player Y, Ball X, Ball Y; Actions: 4
learning_rate = 0.1
discount_factor = 0.95
actions = ["up", "down", "left", "right"]

# Global variables to track previous distances for reward calculation
previous_ball_distance_to_goal = None
previous_player_distance_to_ball = None

def grid_map(value, min_value, max_value, grid_size):
    """Map a value from real coordinates to a grid index."""
    normalized = (value - min_value) / (max_value - min_value)
    index = int(normalized * grid_size)
    return max(0, min(grid_size - 1, index))

def calculate_distance(pos1, pos2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((pos2['x'] - pos1['x']) ** 2 + (pos2['y'] - pos1['y']) ** 2)

previous_ball_distance_to_goal = float('inf')  # Initialize with infinity or a suitable starting value
previous_player_distance_to_ball = float('inf')  # Initialize similarly

def get_reward(player_pos, ball_pos, isGoal, prev_ball_dist_to_goal, prev_player_dist_to_ball):
    """Calculate the reward, avoiding global for previous distances."""
    reward = 0
    goal_pos = {'x': 60, 'y': 0}  # Update based on your field setup

    current_ball_distance_to_goal = calculate_distance(ball_pos, goal_pos)
    current_player_distance_to_ball = calculate_distance(player_pos, ball_pos)

    if isGoal:
        reward += 100
    else:
        # Adjust reward based on distance comparisons
        if current_ball_distance_to_goal < prev_ball_dist_to_goal:
            reward += 10
        if current_player_distance_to_ball < prev_player_dist_to_ball:
            reward += 5
        if current_ball_distance_to_goal > prev_ball_dist_to_goal:
            reward -= 1
        if current_player_distance_to_ball > prev_player_dist_to_ball:
            reward -= 5
        else:
            reward -= 1
    print(reward)
    return reward, current_ball_distance_to_goal, current_player_distance_to_ball

def update_q_table(state, action_index, reward, next_state):
    """Update the Q-table based on the agent's experience."""
    current_q = q_table[state + (action_index,)]
    max_future_q = np.max(q_table[next_state])
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    q_table[state + (action_index,)] = new_q

@app.route("/save")
def save_q_table_to_disk():
    """Save the Q-table to disk."""
    np.save("q_table.npy", q_table)
    return "Q-table saved."

@app.route("/load")
def load_q_table_from_disk():
    """Load the Q-table from disk."""
    global q_table
    if os.path.exists("q_table.npy"):
        q_table = np.load("q_table.npy")
    return "Q-table loaded."

@socketio.on('update_positions')
def handle_update_positions(data):
    """Handle real-time position updates from the game client."""
    global previous_ball_distance_to_goal, previous_player_distance_to_ball
    player_position = data.get('playerPosition')
    ball_position = data.get('ballPosition')
    isGoal = data.get('isGoal', False)

    # Convert positions to grid indices
    player_state_x = grid_map(player_position['x'], -120, 120, 24)
    player_state_y = grid_map(player_position['y'], -70, 70, 14)
    ball_state_x = grid_map(ball_position['x'], -120, 120, 24)
    ball_state_y = grid_map(ball_position['y'], -70, 70, 14)
    state = (player_state_x, player_state_y, ball_state_x, ball_state_y)
    
    # For simplicity, we'll treat the next state as the current state (this should be adapted)
    next_state = state  # In a real scenario, calculate based on the action's outcome

    # Determine action (here we just choose the best known action; consider exploration strategies)
    action_index = np.argmax(q_table[state])
    action = actions[action_index]

    # Calculate reward
    reward, previous_ball_distance_to_goal, previous_player_distance_to_ball = get_reward(
        player_position, ball_position, isGoal,
        previous_ball_distance_to_goal, previous_player_distance_to_ball)

    # Update Q-table
    update_q_table(state, action_index, reward, next_state)

    # Send the next action back to the client
    emit('command', action)

if __name__ == '__main__':
    socketio.run(app)
