from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO
import base64
import numpy as np
import os
from models.DQNAgent import DQNAgent  # Import your DQN agent model
from models.DQNRedAgent import DQNRedAgent  # Import your DQN Red agent model
from threading import Lock

# Initialize Flask application
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize global variables and models
global agent, agentRed
agent = DQNAgent(state_shape=(10, 10, 3), action_size=6)
agentRed = DQNRedAgent(state_shape=(10, 10, 3), action_size=6)

# Initialize state variables
prev_ballPosition = {'x': None, 'y': None}
prev_player_distance_to_ball = 100
prev_opponent_distance_to_ball = 100
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
prev_ballDistanceToGoal = 0
ballDistanceToGoal = 0
blueGoalPos = {'x': -121/2, 'y': 2.44/2, 'z': 0}
redGoalPos = {'x': 121/2, 'y': 2.44/2, 'z': 0}

# Utility function to calculate distance between two points
def calculate_distance(pos1, pos2):
    required_keys = ['x', 'y', 'z']
    for key in required_keys:
        if key not in pos1 or key not in pos2 or pos1[key] is None or pos2[key] is None:
            return 0
    return np.sqrt((pos2['x'] - pos1['x'])**2 + (pos2['y'] - pos1['y'])**2 + (pos2['z'] - pos1['z'])**2)

# Socket.IO event handler for receiving images
@socketio.on('send_image')
def handle_send_image(data):
    """
    Handles the 'send_image' event from the client.

    Args:
        data (dict): A dictionary containing the image and game state data.
    """
    global prev_ballDistanceToGoal, ballDistanceToGoal, current_state, prev_ballPosition, prev_opponent_distance_to_ball, prev_player_distance_to_ball, command_count, done, train, last_command, last_action_blue, last_action_red, multiplayer, start_model_training, isMatch, agent, agentRed
    
    if start_model_training:
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
            if not multiplayer and data['playerPosition'] == {'x': 0, 'y': 0, 'z': 0}:
                agent = agentRed
                player_distance = calculate_distance(data['oppositePlayerPosition'], data['ballPosition'])
                player_commands = get_available_commands(player_distance)
                ballDistanceToGoal = calculate_distance(blueGoalPos, data['ballPosition'])
            else:
                player_distance = calculate_distance(data['playerPosition'], data['ballPosition'])
                player_commands = get_available_commands(player_distance)
                ballDistanceToGoal = calculate_distance(redGoalPos, data['ballPosition'])
            
            if multiplayer:
                opponent_distance = calculate_distance(data['oppositePlayerPosition'], data['ballPosition'])
                opponent_commands = get_available_commands(opponent_distance)
                red_action = get_action(agentRed, next_state, opponent_commands)
            
            blue_action = get_action(agent, next_state, player_commands)
            
            if multiplayer:
                emit('command', {'player': player_commands[blue_action], 'opponent': opponent_commands[red_action]})
            else:
                if data['playerPosition'] == {'x': 0, 'y': 0, 'z': 0}:
                    emit('command', {'player': "", 'opponent': player_commands[blue_action]})
                else:
                    emit('command', {'player': player_commands[blue_action], 'opponent': ""})
            
            if not isMatch:
                update_game_state(data, next_state, goal, scoringSide)

        prev_ballDistanceToGoal = ballDistanceToGoal
        current_state = next_state
        prev_player_distance_to_ball = player_distance
        if multiplayer:
            last_action_red = red_action
            prev_opponent_distance_to_ball = opponent_distance
        last_action_blue = blue_action

# Function to get the action from the agent
def get_action(agent, state, commands):
    """
    Get the action from the agent based on the current state.

    Args:
        agent: The agent model.
        state: The current state of the environment.
        commands (list): List of available commands.

    Returns:
        int: The selected action.
    """
    q_values = agent.model.predict(state)[0]
    if np.random.rand() < agent.epsilon:
        action = np.random.choice(len(commands))
    else:
        action = np.argmax(q_values[:len(commands)])
    return action

# Function to update the game state
def update_game_state(data, next_state, goal, scoringSide):
    """
    Update the game state based on the latest data received.

    Args:
        data (dict): The latest data received.
        next_state: The next state of the environment.
        goal (bool): Whether a goal was scored.
        scoringSide (int): The side that scored.
    """
    global ballDistanceToGoal, prev_ballDistanceToGoal, command_count, done, train, current_state, last_action_red, last_action_blue, multiplayer, prev_opponent_distance_to_ball, prev_player_distance_to_ball, count
    
    if data['playerPosition'] == {'x': 0, 'y': 0, 'z': 0}:
        rewardBlue, done, _ = get_reward(data['oppositePlayerPosition'], data['ballPosition'], goal, scoringSide, prev_player_distance_to_ball, calculate_distance(data['oppositePlayerPosition'], data['ballPosition']), False, ballDistanceToGoal, prev_ballDistanceToGoal)
    else:
        rewardBlue, done, _ = get_reward(data['playerPosition'], data['ballPosition'], goal, scoringSide, prev_player_distance_to_ball, calculate_distance(data['playerPosition'], data['ballPosition']), True, ballDistanceToGoal, prev_ballDistanceToGoal)

    agent.add_experience(current_state, last_action_blue, rewardBlue, next_state, done)

    if multiplayer:
        rewardRed, doneRed, _ = get_reward(data['oppositePlayerPosition'], data['ballPosition'], goal, scoringSide, prev_opponent_distance_to_ball, calculate_distance(data['oppositePlayerPosition'], data['ballPosition']), False, ballDistanceToGoal, prev_ballDistanceToGoal)
        agentRed.add_experience(current_state, last_action_red, rewardRed, next_state, doneRed)
    
    command_count += 1
    if (command_count > 68 or done) and not train:
        train = True
        if multiplayer:
            agentRed.train()
        agent.train()
        emit('reset', True)
        command_count = 0
        done = False
        train = False
    
    if command_count % 5 == 0:
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    if multiplayer:
        if agentRed.epsilon > agentRed.epsilon_min:
            agentRed.epsilon *= agent.epsilon_decay
    if agent.epsilon < agent.epsilon_min:
        agent.epsilon = 1
    
    with model_save_lock:
        agent.save_model()
        if multiplayer:
            agentRed.save_model()

# Function to get the available commands based on the distance to the ball
def get_available_commands(distance):
    """
    Get the available commands based on the distance to the ball.

    Args:
        distance (float): The distance to the ball.

    Returns:
        list: The available commands.
    """
    base_commands = ["up", "down", "left", "right"]
    if distance < 1:
        return base_commands + ["dribble", "shoot"]
    return base_commands

# Function to calculate the reward based on the game state
def get_reward(player_pos, ball_pos, is_goal, scoringSide, prev_ball_distance, current_distance, isBlue, ballDistanceToGoal, prev_ballDistanceToGoal):
    """
    Calculate the reward based on the game state.

    Args:
        player_pos (dict): The player's position.
        ball_pos (dict): The ball's position.
        is_goal (bool): Whether a goal was scored.
        scoringSide (int): The side that scored.
        prev_ball_distance (float): The previous distance to the ball.
        current_distance (float): The current distance to the ball.
        isBlue (bool): Whether the player is on the blue team.
        ballDistanceToGoal (float): The current distance from the ball to the goal.
        prev_ballDistanceToGoal (float): The previous distance from the ball to the goal.

    Returns:
        tuple: The reward, whether the game is done, and the current distance to the ball.
    """
    reward = 0
    if is_goal:
        if isBlue:
            reward += 100 * scoringSide
        else:
            reward -= 100 * scoringSide
    else:
        if ballDistanceToGoal < prev_ballDistanceToGoal:
            reward += 25
        elif ballDistanceToGoal > prev_ballDistanceToGoal:
            reward -= 5

        if current_distance < prev_ball_distance:
            reward += 5
        elif current_distance > prev_ball_distance:
            reward -= 10
        else:
            reward -= 10
    
    return reward, is_goal, current_distance

# Valid model names for upload and download
VALID_MODEL_NAMES = ['dqn_model_red.keras', 'dqn_model.keras']

# Route for uploading a model
@app.route("/upload", methods=['POST'])
def upload_model_route():
    """
    Route for uploading a model.

    Returns:
        Response: A JSON response indicating the result of the upload.
    """
    model_file = request.files['file']
    model_name = model_file.filename
    
    if model_name in VALID_MODEL_NAMES:
        model_path = os.path.join(model_name)
        model_file.save(model_path)
        return jsonify({"message": f"Model {model_name} uploaded successfully."})
    else:
        return jsonify({"message": "Invalid file name. Please upload either dqn_model_red.keras or dqn_model.keras."}), 400

# Route for downloading a model
@app.route("/download", methods=['GET'])
def download_model_route():
    """
    Route for downloading a model.

    Returns:
        Response: The model file or a JSON response indicating an error.
    """
    model_name = request.args.get('model_name')

    if model_name in VALID_MODEL_NAMES:
        model_path = os.path.join(model_name)
        if os.path.exists(model_path):
            return send_file(model_path, as_attachment=True)
        else:
            return jsonify({"message": "Model not found."}), 404
    else:
        return jsonify({"message": "Invalid model name. Please request either dqn_model_red.keras or dqn_model.keras."}), 400

# Route for starting the training
@app.route('/start', methods=['POST'])
def start_training():
    """
    Route for starting the training.

    Returns:
        Response: A JSON response indicating the result of the start request.
    """
    global start_model_training, isMatch, agent, agentRed
    data = request.get_json()
    model = data.get('model')
    
    if model is None:
        return jsonify({"error": "No model specified"}), 400
    
    if model == "q-learning":
        agent.epsilon = 1
        agentRed.epsilon = 1
        isMatch = False
    else:
        agentRed.epsilon = 0.1
        agent.epsilon = 0.1
        isMatch = True
    
    start_model_training = True

    return jsonify({"message": "Training started."})

# Route for stopping the training
@app.route('/stop', methods=['POST'])
def stop_training():
    """
    Route for stopping the training.

    Returns:
        Response: A JSON response indicating the result of the stop request.
    """
    global start_model_training, command_count, done, agent, agentRed, multiplayer
    start_model_training = False
    done = False
    command_count = 0
    socketio.emit('reset', True)
    agent.epsilon = 1.0
    agent = DQNAgent(state_shape=(10, 10, 3), action_size=6)
    if multiplayer:
        agentRed.epsilon = 1.0
    return jsonify({"message": "Training stopped and model reset."})

# Function to preprocess the image
def preprocess_image(image_base64, target_size=(10, 10)):
    """
    Preprocess the image from base64 to a format suitable for the model.

    Args:
        image_base64 (str): The base64 encoded image.
        target_size (tuple): The target size of the image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    image = load_img(BytesIO(image_bytes), target_size=target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if __name__ == '__main__':
    socketio.run(app, debug=True)
