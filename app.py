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
agent = DQNAgent(state_shape=(10, 10, 3), action_size=4)

prev_ballPosition = {'x': None, 'y': None}
prev_playerPosition = {'x': None, 'y': None}

def calculate_distance(pos1, pos2):
    """Calculate the Euclidean distance between two points."""
    if(pos1['x'] == None or pos2['x'] == None or pos1['y'] == None or pos2['y'] == None):
        return 0
    return np.sqrt((pos2['x'] - pos1['x'])**2 + (pos2['y'] - pos1['y'])**2)

model_save_lock = Lock()
global current_state  # Ensure this is defined outside your functions to maintain state across calls
current_state = None

@socketio.on('send_image')
def handle_send_image(data):
    global current_state, prev_ballPosition, prev_playerPosition
    # Basic validation to ensure required data is present
    if not all(key in data for key in ['image', 'playerPosition', 'ballPosition', 'isGoal']):
        print("Missing data in request.")
        return

    # Preprocess the current image to get the current state
    next_state = preprocess_image(data['image'])
    if current_state is not None:
        
        # Assuming you have a way to determine the action, reward, and if it's done
        action = np.argmax(agent.model.predict(current_state)[0])
        # Emit the command before calculating the reward
        commands = ["up", "down", "left", "right"]
        command = commands[action]
        print(command)
        
        emit('command', command)  # Client should perform the action and send back the next image

        # Calculate reward and whether the episode is done based on the action
        prev_distance = calculate_distance(prev_playerPosition, prev_ballPosition)
        current_distance = calculate_distance(data['playerPosition'], data['ballPosition'])
        reward, done, _ = get_reward(data['playerPosition'], data['ballPosition'], data['isGoal']['intersecting'], prev_distance, current_distance)
        print(reward)
        # Update replay memory and train the model
        agent.add_experience(current_state, action, reward, next_state, done)

    # Update state for the next iteration
    current_state = next_state

    # Optionally, perform training and model saving here
    agent.train()
    with model_save_lock:
        agent.save_model()

    # Update previous positions for the next call
    prev_ballPosition, prev_playerPosition = data['ballPosition'], data['playerPosition']

def get_reward(player_pos, ball_pos, isGoal, prev_ball_distance, current_distance):
    """Calculate the reward."""
    reward = 0
    if isGoal:
        reward += 100
    elif current_distance < prev_ball_distance:
        reward += 5  # Moved closer to the ball
    else:
        reward -= 10  # Moved away or stayed the same distance
    return reward, isGoal, prev_ball_distance  # Returning current_distance for clarity

@app.route("/save", methods=['POST'])
def save_model_route():
    agent.save_model()
    return jsonify({"message": "Model saved successfully."})

@app.route("/load", methods=['GET'])
def load_model_route():
    global agent
    agent.load_model()  # Ensure DQNAgent has a load_model method or adjust this line
    return jsonify({"message": "Model loaded successfully."})

def preprocess_image(image_base64, target_size=(10, 10)):
    """
    Convert a base64-encoded image to a normalized array suitable for DQN model input.
    
    Parameters:
        image_base64 (str): The base64-encoded representation of the image.
        target_size (tuple): The desired size of the output image array (height, width).
    
    Returns:
        A numpy array representing the preprocessed image.
    """
    # Decode the base64 string
    image_bytes = base64.b64decode(image_base64.split(",")[1])
    
    # Load the image from bytes and resize it
    image = load_img(BytesIO(image_bytes), target_size=target_size)
    
    # Convert the PIL image to a numpy array and normalize pixel values to [0, 1]
    image_array = img_to_array(image) / 255.0
    
    # Expand dimensions to include a batch size of 1
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

if __name__ == '__main__':
    socketio.run(app, debug=True)
