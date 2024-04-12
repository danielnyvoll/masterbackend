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


current_state, command_count, done = None, 0, False
training_session_active = False

def preprocess_image(image_base64, target_size=(10, 10)):
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    image = load_img(BytesIO(image_bytes), target_size=target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def get_reward(player_pos, ball_pos, isGoal, prev_ball_distance, current_distance):
    reward = 0
    if isGoal:
        reward += 100  # Reward for scoring a goal
    elif current_distance < prev_ball_distance:
        reward += 5  # Reward for moving closer
    elif current_distance == prev_ball_distance:
        reward -= 20  # Penalty for standing still
    else:
        reward -= 10  # Penalty for moving away
    return reward, isGoal, current_distance

@app.route('/start', methods=['POST'])
def start_model():
    global training_session_active
    data = request.get_json()
    print(agent.epsilon)
    model_selection = data.get('model')
    try:
        initialize_model(model_selection)
        training_session_active = True
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
    global current_state, prev_playerPosition, prev_ballPosition, command_count, done, training_session_active
    if not training_session_active:
        emit('reset', True)
    if not training_session_active or not all(key in data for key in ['image', 'playerPosition', 'ballPosition', 'isGoal']):
        return
    print("Hit")
    next_state = preprocess_image(data['image'])
    current_distance = calculate_distance(data['playerPosition'], data['ballPosition'])
    commands = ["up", "down", "left", "right", "dribble", "shoot"] if current_distance < 1 else ["up", "down", "left", "right"]
    if current_state is None or done:
        current_state = next_state
    
    take_action(commands, current_state, next_state, data['playerPosition'], data['ballPosition'], data['isGoal']['intersecting'])
    
    if command_count > 25 or done:
        emit('next', True)
        reset_training()
    else:
        prev_playerPosition = data['playerPosition']
        prev_ballPosition = data['ballPosition']
        current_state = next_state

def take_action(commands, current_state, next_state, playerPosition, ballPosition, isGoal):
    global command_count, done
    q_values = agent.model.predict(current_state)[0]
    action = np.random.choice(len(commands)) if np.random.rand() <= agent.epsilon else np.argmax(q_values) % len(commands)
    command = commands[action]
    emit('command', command)
    reward, done, _ = get_reward(playerPosition, ballPosition, isGoal, calculate_distance(prev_playerPosition, prev_ballPosition), calculate_distance(playerPosition, ballPosition))
    emit('reward', reward)
    agent.add_experience(current_state, action, reward, next_state, done)
    command_count += 1

def reset_training():
    global command_count, done, current_state
    emit('reset', True)
    command_count = 0
    done = False
    current_state = None
    with model_save_lock:
        agent.save_model()

if __name__ == '__main__':
    socketio.run(app, debug=True)
