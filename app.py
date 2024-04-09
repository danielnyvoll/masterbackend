from flask import Flask, jsonify
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

# Initialize the DQN Agent
agent = DQNAgent(state_shape=(10, 10, 3), action_size=6)

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
global current_state, command_count, done
current_state = None
command_count = 0
done = False

@socketio.on('send_image')
def handle_send_image(data):
    global current_state, prev_ballPosition, prev_playerPosition, command_count, done
    
    if not all(key in data for key in ['image', 'playerPosition', 'ballPosition', 'isGoal']):
        print("Missing data in request.")
        return

    next_state = preprocess_image(data['image'])
    
    # Calculate distance to the ball
    current_distance = calculate_distance(data['playerPosition'], data['ballPosition'])

    # Determine the list of possible actions
    commands = ["up", "down", "left", "right"]
    if current_distance < 1:  # Player is close enough to shoot or dribble
        commands.extend(["dribble", "shoot"])

    if current_state is not None:
        q_values = agent.model.predict(current_state)[0]
        print("Q-values:", q_values)
        
        if np.random.rand() <= agent.epsilon:
            action = np.random.choice(len(commands))  # Explore
        else:
            action = np.argmax(q_values)  # Use the Q-values to pick the best action
            action = action % len(commands)  # Ensure action index is within the range of available commands
        
        command = commands[action]
        emit('command', command)  # Emit command to the client
        
        command_count += 1
        prev_distance = calculate_distance(prev_playerPosition, prev_ballPosition)
        
        reward, done, _ = get_reward(data['playerPosition'], data['ballPosition'], data['isGoal']['intersecting'], prev_distance, current_distance)
        print("Epsilon: ", agent.epsilon)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        agent.add_experience(current_state, action, reward, next_state, done)
        print(f"Command: {command}, Reward: {reward}, Command Count: {command_count}")

    if command_count > 25 or done:
        emit('reset', True)  # Reset signal to client
        command_count = 0
        done = False  # Reset done for the next episode

    current_state = next_state
    agent.train()  # Train the agent with experiences gathered

    with model_save_lock:
        print("Save!!")
        agent.save_model()  # Specify your model path and name

    prev_ballPosition, prev_playerPosition = data['ballPosition'], data['playerPosition']


def get_reward(player_pos, ball_pos, isGoal, prev_ball_distance, current_distance):
    """Calculate the reward with the assumption of any goal being valid.
    
    Increases the penalty for standing still compared to moving further away from the ball,
    to discourage the player from getting stuck in the wall.
    """
    reward = 0
    print("Current distance from the ball:", current_distance)
    print("Previous distance from the ball:", prev_ball_distance)

    if isGoal:
        reward += 100  # Reward for scoring a goal
    elif current_distance < prev_ball_distance:
        reward += 5  # Reward for moving closer to the ball
    elif current_distance == prev_ball_distance:
        reward -= 20  # Increased penalty for standing still
    else:
        reward -= 10  # Penalty for moving further away from the ball
    
    return reward, isGoal, current_distance


@app.route("/save", methods=['POST'])
def save_model_route():
    agent.save_model(os.path.join('model_path', 'model_name.h5'))  # Adjust with your model path and name
    return jsonify({"message": "Model saved successfully."})

@app.route("/load", methods=['GET'])
def load_model_route():
    agent.load_model(os.path.join('model_path', 'model_name.h5'))  # Adjust with your model path and name
    return jsonify({"message": "Model loaded successfully."})

def preprocess_image(image_base64, target_size=(10, 10)):
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    image = load_img(BytesIO(image_bytes), target_size=target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if __name__ == '__main__':
    socketio.run(app, debug=True)
