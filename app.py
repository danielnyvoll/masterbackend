from flask import Flask, jsonify
from flask_socketio import SocketIO
from threading import Lock, Event
from main import start_simulation, get_simulation_data, pause_simulation, stop_simulation

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
simulation_running = False
simulation_data = {}

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/start")
def start():
    global simulation_running
    if not simulation_running:
        simulation_running = True
        socketio.start_background_task(background_simulation_task)
        return "<p>Simulation Started</p>"
    else:
        return "<p>Simulation is already running</p>"

@app.route("/stop")
def stop():
    global simulation_running
    simulation_running = False
    return "<p>Simulation Stopped</p>"

@app.route("/pause")
def pause():
    pause_simulation()
    return "<p>Simulation Paused</p>"

def start_simulation(socketio_instance):
    global simulation_running
    while simulation_running:
            background_task = socketio.start_background_task(background_simulation_task)


def background_simulation_task():
    global simulation_data
    start_simulation(socketio)
    print("Exiting background task")

@socketio.on('connect')
def on_connect():
    print('Client connected')
    simulation_data = get_simulation_data()
    socketio.emit('simulation_data', simulation_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, debug=True)
