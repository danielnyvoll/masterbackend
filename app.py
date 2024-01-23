from flask import Flask, jsonify
from flask_socketio import SocketIO
from threading import Lock
from main import start_simulation, get_simulation_data

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

thread = None
thread_lock = Lock()

# Shared data structure
simulation_data = {}
simulation_data_lock = Lock()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/start")
def start():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_simulation_task)
    return "<p>Simulation Started</p>"

@app.route("/data")
def get_data():
    with simulation_data_lock:
        return jsonify(simulation_data)

@socketio.on('connect')
def on_connect():
    print('Client connected')
    # Emit test data
    simulation_data = get_simulation_data()
    socketio.emit('simulation_data', simulation_data)


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def background_simulation_task():
    with app.app_context():
        # Pass the shared data structure and its lock to the simulation
        start_simulation(socketio, simulation_data, simulation_data_lock)

if __name__ == '__main__':
    socketio.run(app, debug=True)
