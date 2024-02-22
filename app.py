from flask import Flask
from flask_socketio import SocketIO, emit
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import os
import numpy as np

# Import your model classes
#from models.q_learning import QLearningModel
from models.deep_q_learning import DeepQLearningModel

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)
previous_ball_distance_to_goal = float('inf')
previous_player_distance_to_ball = float('inf')
command_mapping = {
    0: "up",
    1: "left",
    2: "down",
    3: "right"
}

# Placeholder for the currently selected model
current_model = DeepQLearningModel(state_size=96, action_size=4, learning_rate=0.001)  # Placeholder parameters

@app.route('/select_model', methods=['POST'])
def select_model():
    global current_model
    data = request.json
    model_choice = data.get('model_choice', 'q_learning').lower()

    #if model_choice == "q_learning":
      #  current_model = QLearningModel(state_size=(24, 14, 24, 14), action_size=4, learning_rate=0.1, discount_factor=0.95)
    if model_choice == "deep_q_learning":
        current_model = DeepQLearningModel(state_size=96, action_size=4, learning_rate=0.001)  # Placeholder parameters
    else:
        return jsonify({"error": "Invalid model choice specified"}), 400
    
    return jsonify({"message": "Model selected: {}".format(model_choice)}), 200


def calculate_distance(pos1, pos2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((pos2['x'] - pos1['x']) ** 2 + (pos2['y'] - pos1['y']) ** 2)

def get_reward(player_pos, ball_pos, isGoal, prev_ball_dist_to_goal, prev_player_dist_to_ball):
    goal_pos = {'x': 60, 'y': 0}  # Assuming the goal's position for simplicity

    current_ball_distance_to_goal = calculate_distance(ball_pos, goal_pos)
    current_player_distance_to_ball = calculate_distance(player_pos, ball_pos)

    reward = 0
    reward += prev_ball_dist_to_goal - current_ball_distance_to_goal
    reward += prev_player_dist_to_ball - current_player_distance_to_ball 

    return reward, current_ball_distance_to_goal, current_player_distance_to_ball


@socketio.on('update_positions')
def handle_update_positions(data):
    global current_model, previous_ball_distance_to_goal, previous_player_distance_to_ball

    if not current_model:
        emit('error', {'message': 'Model not selected'})
        return

    player_position = data.get('playerPosition')
    ball_position = data.get('ballPosition')
    isGoal = data.get('intersecting')
    print(isGoal)
    # Convert positions to the expected format for distance calculation
    player_pos = {'x': player_position['x'], 'y': player_position['y']}
    ball_pos = {'x': ball_position['x'], 'y': ball_position['y']}

    reward, new_ball_distance_to_goal, new_player_distance_to_ball = get_reward(
        player_pos, ball_pos, isGoal, 
        previous_ball_distance_to_goal, previous_player_distance_to_ball)

    # Update previous distances for the next call
    previous_ball_distance_to_goal = new_ball_distance_to_goal
    previous_player_distance_to_ball = new_player_distance_to_ball

    # Prepare the state for the model
    state = np.array([player_pos['x'], player_pos['y'], ball_pos['x'], ball_pos['y']])
    state = np.reshape(state, [1, 4])

    action = current_model.choose_action(state)

    # Emit the chosen action
    emit('command', command_mapping.get(action, "Unknown"))

if __name__ == '__main__':
    socketio.run(app, debug=True)
