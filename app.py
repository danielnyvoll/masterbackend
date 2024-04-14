from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import base64
import numpy as np
import os
from models.DQNAgent import DQNAgent
from threading import Lock

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the DQN Agent with a default model.
agent = DQNAgent(state_shape=(10, 10, 3), action_size=6, model_file='dqn_model.keras')

# For simplicity, using a single function to handle distance calculations.
def calculate_distance(pos1, pos2):
    if None in (pos1['x'], pos2['x'], pos1['y'], pos2['y']):
        return 0
    return np.sqrt((pos2['x'] - pos1['x'])**2 + (pos2['y'] - pos1['y'])**2)

model_save_lock = Lock()

prev_playerPosition = {'x': None, 'y': None}
prev_ballPosition = {'x': None, 'y': None}

blueSideGoal = {'x': -121/2, 'y': 2.44/2, 'z': 0}
redSideGoal ={'x': 121/2, 'y': 2.44/2, 'z': 0}

current_state, command_count, done = {}, 0, False
training_session_active = False

def preprocess_image(image_base64, target_size=(10, 10)):
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    image = load_img(BytesIO(image_bytes), target_size=target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def get_reward(player_pos, ball_pos, isGoal, prev_ball_distance, current_distance, player_team):
    reward = 0
    goal_scored = isGoal
    
    if goal_scored:
            reward += 100  # Reward for scoring a goal

    elif current_distance < prev_ball_distance:
        reward += 5  # Reward for moving closer to the ball
    elif current_distance == prev_ball_distance:
        reward -= 20  # Penalty for standing still
    else:
        reward -= 10  # Penalty for moving away from the ball
    
    return reward, goal_scored, current_distance



@app.route('/start', methods=['POST'])
def start_model():
    data = request.get_json()
    model_selection = data.get('model')
    try:
        initialize_model(model_selection)
        return jsonify({"message": f"Model training started with {model_selection}."})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/stop', methods=['POST'])
def stop_model():
    global training_session_active, command_count
    training_session_active = False
    command_count = 0
    agent.save_model()
    return jsonify({"message": "Model training stopped."})

def initialize_model(model_selection):
    global training_session_active

    # Adjust agent initialization based on the selected model.
    if model_selection in ['q-deep-learning', 'q-learning']:
        print(f"{model_selection} Model selected.")
        training_session_active = True
    else:
        print("Default Model selected.")

def save_model():
    with model_save_lock:
        # Specify your model path and name here.
        model_path = os.path.join('model_path', 'model_name.h5')
        print(f"Saving model to {model_path}")
        agent.save_model(model_path)

@app.route("/save", methods=['POST'])
def save_model_route():
    save_model()
    return jsonify({"message": "Model saved successfully."})

@app.route("/load", methods=['GET'])
def load_model_route():
    model_path = os.path.join('model_path', 'model_name.h5')
    agent.load_model(model_path)
    return jsonify({"message": "Model loaded successfully."})

@socketio.on('send_image')
def handle_send_image(data):
    global current_state, command_count, done, training_session_active
    if not training_session_active:
        emit('reset', True)
        return
    if not all(key in data for key in ['image', 'gameState', 'ballPosition', 'isGoal']):
        print("Missing required data")
        return

    if current_state is None or done:
        current_state = {}

    players_data = data.get('gameState', {}).get('players', {})
    if 'image' in data:
        image_state = preprocess_image(data['image'])

    for player_id, player_info in players_data.items():
        if not player_info:
            print(f"Missing player information for player ID {player_id}")
            continue
        
        player_team = player_info['team']
        player_position = {'x': player_info['x'], 'y': player_info['y'], 'z': player_info['z']}
        
        current_state[player_id] = image_state  # Assigning processed image to each player's state

        current_distance = calculate_distance(player_position, data['ballPosition'])
        commands = ["up", "down", "left", "right", "dribble", "shoot"] if current_distance < 1 else ["up", "down", "left", "right"]
        take_action(player_id, player_team, commands, current_state[player_id], player_position, data['ballPosition'], data['isGoal'])

    if command_count > 25 or done:
        emit('next', True)
        reset_training()
    else:
        command_count += 1

def take_action(player_id, player_team, commands, current_state, playerPosition, ballPosition, isGoal):
    global command_count, done, prev_playerPosition, prev_ballPosition

    if player_id not in prev_playerPosition:
        prev_playerPosition[player_id] = {'x': None, 'y': None}
    if player_id not in prev_ballPosition:
        prev_ballPosition[player_id] = {'x': None, 'y': None}

    q_values = agent.model.predict(current_state)[0]
    action = np.argmax(q_values) % len(commands) if player_team == 'red' else np.random.choice(len(commands))

    command = commands[action]
    emit('command', {'playerId': player_id, 'command': command})

    prev_ball_distance = calculate_distance(prev_playerPosition[player_id], prev_ballPosition[player_id])
    current_distance = calculate_distance(playerPosition, ballPosition)
    reward, done, _ = get_reward(playerPosition, ballPosition, isGoal, prev_ball_distance, current_distance, player_team)
    emit('reward', {'id': player_id, 'reward': reward, 'command': command})

    agent.add_experience(current_state, action, reward, current_state, done)

    prev_playerPosition[player_id] = playerPosition
    prev_ballPosition[player_id] = ballPosition

    if command_count > 25 or done:
        reset_training()

def reset_training():
    global command_count, done, current_state
    emit('reset', True)
    command_count = 0
    done = False
    current_state = {}
    with model_save_lock:
        agent.save_model()
if __name__ == '__main__':
    socketio.run(app, debug=True)
