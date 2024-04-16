from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import base64
import numpy as np
import os
from models.DQNAgent import DQNAgent
from models.DQNRedAgent import DQNRedAgent
from threading import Lock

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize the DQN Agent
agent = DQNAgent(state_shape=(10, 10, 3), action_size=6)
agentRed = DQNRedAgent(state_shape=(10, 10, 3), action_size=6)

prev_ballPosition = {'x': None, 'y': None}
prev_playerPosition = {'x': None, 'y': None}

# For simplicity, using a single function to handle distance calculations
def calculate_distance(pos1, pos2):
    """Calculate the Euclidean distance between two points."""
    if pos1['x'] is None or pos2['x'] is None or pos1['y'] is None or pos2['y'] is None:
        return 0
    return np.sqrt((pos2['x'] - pos1['x'])**2 + (pos2['y'] - pos1['y'])**2)

model_save_lock = Lock()

# Ensure these are defined outside your functions to maintain state across calls
global current_state, command_count, done, train
start_model_training = False
current_state = None
command_count = 0
done = False
train = False
last_command = None
last_action_red = 0
last_action_blue = 0
multiplayer = False

@socketio.on('send_image')
def handle_send_image(data):
    global current_state, prev_ballPosition, prev_playerPosition, command_count, done, train, last_command, last_action_blue, last_action_red, multiplayer, start_model_training
    if(start_model_training):
        if train:
            return

        if not all(key in data for key in ['image', 'playerPosition', 'oppositePlayerPosition', 'ballPosition', 'isGoal']):
            print("Missing data in request.")
            return
        multiplayer = data['isMultiplayer']['multiplayer']
        next_state = preprocess_image(data['image'])
        if current_state is not None:
            
            if(multiplayer):
                opponent_distance = calculate_distance(data['oppositePlayerPosition'], data['ballPosition'])
                opponent_commands = get_available_commands(opponent_distance)
                red_action = get_action(agentRed, next_state, opponent_commands)  # Adjustable epsilon

            player_distance = calculate_distance(data['playerPosition'], data['ballPosition'])

            # Define commands based on distances for both players
            player_commands = get_available_commands(player_distance)

            # Fetch commands for both players
            blue_action = get_action(agent, next_state, player_commands)  
            if(multiplayer):
                emit('command', {'player': player_commands[blue_action], 'opponent': opponent_commands[red_action]})
            else:
                emit('command', {'player': player_commands[blue_action], 'opponent': ""})

            update_game_state(data, next_state)


        current_state = next_state
        if(multiplayer):
            last_action_red = red_action
        last_action_blue = blue_action

def get_action(agent, state, commands ):
    q_values = agent.model.predict(state)[0]
    if np.random.rand() < agent.epsilon:
        # Exploration
        action = np.random.choice(len(commands))
    else:
        # Exploitation
        action = np.argmax(q_values[:len(commands)])
    return action

def update_game_state(data, next_state):
    global command_count, done, train, current_state, last_action_red, last_action_blue, multiplayer
    prev_distance = calculate_distance(prev_playerPosition, prev_ballPosition)
    
    rewardBlue, done, _ = get_reward(data['playerPosition'], data['ballPosition'], data['isGoal']['intersecting'], prev_distance, calculate_distance(data['playerPosition'], data['ballPosition']))
    if(multiplayer):
        rewardRed, doneRed, _ = get_reward(data['oppositePlayerPosition'], data['ballPosition'], data['isGoal']['intersecting'], prev_distance, calculate_distance(data['oppositePlayerPosition'], data['ballPosition']))
        agentRed.add_experience(current_state, last_action_red, rewardRed, next_state, doneRed)

    # Update experiences and training
    agent.add_experience(current_state, last_action_blue, rewardBlue, next_state, done)

    

    command_count += 1
    print(command_count)
    print(done)
    if (command_count > 38 or done) and not train:
        train = True
        if(multiplayer):
            agentRed.train()
        agent.train()
        emit('reset', True)
        command_count = 0
        done = False
        train = False

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    if(multiplayer):
        if agentRed.epsilon > agentRed.epsilon_min:
            agentRed.epsilon *= agent.epsilon_decay

    with model_save_lock:
        agent.save_model()
        if(multiplayer):
            agentRed.save_model()
    

def get_available_commands(distance):
    base_commands = ["up", "down", "left", "right"]
    if distance < 1:
        return base_commands + ["dribble", "shoot"]
    return base_commands

def get_reward(player_pos, ball_pos, is_goal, prev_ball_distance, current_distance):
    reward = 0
    if is_goal:
        reward += 100
    elif current_distance < prev_ball_distance:
        reward += 5
    elif current_distance == prev_ball_distance:
        reward -= 20
    else:
        reward -= 10
    return reward, is_goal, current_distance




@app.route("/save", methods=['POST'])
def save_model_route():
    agent.save_model(os.path.join('model_path', 'model_name.h5'))  # Adjust with your model path and name
    return jsonify({"message": "Model saved successfully."})

@app.route("/load", methods=['GET'])
def load_model_route():
    agent.load_model(os.path.join('model_path', 'model_name.h5'))  # Adjust with your model path and name
    return jsonify({"message": "Model loaded successfully."})

@app.route('/start', methods=['POST'])
def start_training():
    global start_model_training
    start_model_training = True
    return jsonify({"message": "Training started."})

@app.route('/stop', methods=['POST'])
def stop_training():
    global start_model_training, command_count, done, agent, agentRed, multiplayer
    start_model_training = False
    done = False
    command_count = 0
    socketio.emit('reset',  True)  # Emit reset to the frontend
    agent.epsilon = 1.0  # Reset epsilon after training stops
    if multiplayer:
        agentRed.epsilon = 1.0
    return jsonify({"message": "Training stopped and model reset."})

def preprocess_image(image_base64, target_size=(10, 10)):
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    image = load_img(BytesIO(image_bytes), target_size=target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array



if __name__ == '__main__':
    socketio.run(app, debug=True)