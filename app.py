from flask import Flask, jsonify, request, send_file
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
global agent, agentRed
agent = DQNAgent(state_shape=(10, 10, 3), action_size=6)
agentRed = DQNRedAgent(state_shape=(10, 10, 3), action_size=6)

prev_ballPosition = {'x': None, 'y': None}
prev_player_distance_to_ball = 100
prev_opponent_distance_to_ball = 100

def calculate_distance(pos1, pos2):
    required_keys = ['x', 'y', 'z']
    for key in required_keys:
        if key not in pos1 or key not in pos2 or pos1[key] is None or pos2[key] is None:
            return 0
    return np.sqrt((pos2['x'] - pos1['x'])**2 + (pos2['y'] - pos1['y'])**2 + (pos2['z'] - pos1['z'])**2)

model_save_lock = Lock()

global current_state, command_count, done, train, isMatch
start_model_training = False
current_state = None
command_count = 0
done = False
train = False
last_command = None
last_action_red = 0
last_action_blue = 0
multiplayer = False
isMatch = False

@socketio.on('send_image')
def handle_send_image(data):
    global current_state, prev_ballPosition, prev_opponent_distance_to_ball, prev_player_distance_to_ball, command_count, done, train, last_command, last_action_blue, last_action_red, multiplayer, start_model_training, isMatch, agent, agentRed
    if(start_model_training):
        if train:
            return

        if not all(key in data for key in ['image', 'playerPosition', 'oppositePlayerPosition', 'ballPosition', 'isGoal']):
            print("Missing data in request.")
            return
        multiplayer = data['isMultiplayer']['multiplayer']
        next_state = preprocess_image(data['image'])
        isGoal = data['isGoal']
        goal = isGoal['intersecting']
        scoringSide = isGoal['scoringSide']
        if current_state is not None:
            if not multiplayer and data['playerPosition'] ==  {'x': 0, 'y': 0, 'z': 0}:
                agent = agentRed
            if(multiplayer):
                opponent_distance = calculate_distance(data['oppositePlayerPosition'], data['ballPosition'])
                opponent_commands = get_available_commands(opponent_distance)
                red_action = get_action(agentRed, next_state, opponent_commands)

            player_distance = calculate_distance(data['playerPosition'], data['ballPosition'])
            print(player_distance)
            player_commands = get_available_commands(player_distance)

            blue_action = get_action(agent, next_state, player_commands)  
            if(multiplayer):
                emit('command', {'player': player_commands[blue_action], 'opponent': opponent_commands[red_action]})
            else:
                emit('command', {'player': player_commands[blue_action], 'opponent': ""})
            if not isMatch:
                update_game_state(data, next_state, goal, scoringSide)


        current_state = next_state
        prev_player_distance_to_ball = player_distance
        if(multiplayer):
            last_action_red = red_action
            prev_opponent_distance_to_ball = opponent_distance
        last_action_blue = blue_action

def get_action(agent, state, commands ):
    q_values = agent.model.predict(state)[0]
    if np.random.rand() < agent.epsilon:
        action = np.random.choice(len(commands))
    else:
        action = np.argmax(q_values[:len(commands)])
    return action
count = 0
def update_game_state(data, next_state, goal, scoringSide):
    global command_count, done, train, current_state, last_action_red, last_action_blue, multiplayer, prev_opponent_distance_to_ball, prev_player_distance_to_ball, count
    
    rewardBlue, done, _ = get_reward(data['playerPosition'], data['ballPosition'], goal, scoringSide, prev_player_distance_to_ball, calculate_distance(data['playerPosition'], data['ballPosition']), True)

    agent.add_experience(current_state, last_action_blue, rewardBlue, next_state, done)

    if multiplayer:
        rewardRed, doneRed, _ = get_reward(data['oppositePlayerPosition'], data['ballPosition'], goal, scoringSide, prev_opponent_distance_to_ball, calculate_distance(data['oppositePlayerPosition'], data['ballPosition']), False)
        agentRed.add_experience(current_state, last_action_red, rewardRed, next_state, doneRed)

    command_count += 1
    if (command_count > 38 or done) and not train:
        train = True
        if(multiplayer):
            agentRed.train()
        agent.train()
        count+=1
        print(count)
        emit('reset', True)
        command_count = 0
        done = False
        train = False

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    if(multiplayer):
        if agentRed.epsilon > agentRed.epsilon_min:
            agentRed.epsilon *= agent.epsilon_decay
    if agent.epsilon < agent.epsilon_min:
        agent.epsilon = 1
    with model_save_lock:
        agent.save_model()
        if(multiplayer):
            agentRed.save_model()
    

def get_available_commands(distance):
    base_commands = ["up", "down", "left", "right"]
    if distance < 1:
        return base_commands + ["dribble", "shoot"]
    return base_commands

def get_reward(player_pos, ball_pos, is_goal, scoringSide, prev_ball_distance, current_distance, isBlue):
    reward = 0
    if is_goal:
        if (isBlue):
            reward += 100 * scoringSide
        else:
            reward -= 100 * scoringSide
    elif current_distance < prev_ball_distance:
        reward += 5
    elif current_distance == prev_ball_distance:
        reward -= 20
    else:
        reward -= 10
    return reward, is_goal, current_distance

VALID_MODEL_NAMES = ['dqn_model_red.keras', 'dqn_model.keras']

@app.route("/upload", methods=['POST'])
def upload_model_route():
    model_file = request.files['file'] 
    model_name = model_file.filename 
    
    if model_name in VALID_MODEL_NAMES:
        model_path = os.path.join(model_name)
        model_file.save(model_path)  
        return jsonify({"message": f"Model {model_name} uploaded successfully."})
    else:
        return jsonify({"message": "Invalid file name. Please upload either dqn_model_red.keras or dqn_model.keras."}), 400

@app.route("/download", methods=['GET'])
def download_model_route():
    model_name = request.args.get('model_name') 

    if model_name in VALID_MODEL_NAMES:
        model_path = os.path.join(model_name) 
        if os.path.exists(model_path):
            return send_file(model_path, as_attachment=True)
        else:
            return jsonify({"message": "Model not found."}), 404
    else:
        return jsonify({"message": "Invalid model name. Please request either dqn_model_red.keras or dqn_model.keras."}), 400

@app.route('/start', methods=['POST'])
def start_training():
    global start_model_training, isMatch, agent, agentRed
    data = request.get_json()  
    model = data.get('model') 
    if model is None:
        return jsonify({"error": "No model specified"}), 400
    if(model == "q-learning"):
        agent.epsilon = 1
        agentRed.epsilon = 1
        isMatch = False
    else:
        agentRed.epsilon = 0.1
        agent.epsilon = 0.1
        isMatch = True
    start_model_training = True

    return jsonify({"message": "Training started."})

@app.route('/stop', methods=['POST'])
def stop_training():
    global start_model_training, command_count, done, agent, agentRed, multiplayer
    start_model_training = False
    done = False
    command_count = 0
    socketio.emit('reset',  True)  
    agent.epsilon = 1.0 
    agent = DQNAgent(state_shape=(10, 10, 3), action_size=6)
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
