from threading import Lock
from values import *
from tools import Pose
from ball import Ball
from simulation import *
from state import FiniteStateMachineBall, MoveForwardStateBall, Reflection
from environment import Environment
import datetime
import os
import json

# ______________________________________________________________________________
# simulation2D function
def simulation2D(players, shockable = True, full_vision = True):
    """
    This function initialize the simulation and return a object that the user 
    can pass the controls and get the sensors information.

    :param players: a list of Players for simulation
    :type: list of Player
    :param shockable: parameter that informs if players will collide with themselves
    :type shockable: bool
    :param full_vision: parameter that informs if player will see every thing even if it’s not in the vision cone.
    :type full_vision: bool
    """
    behavierBall = FiniteStateMachineBall(MoveForwardStateBall(True))
    poseBall = Pose(PIX2M * SCREEN_WIDTH*1/4.0, PIX2M * SCREEN_HEIGHT / 2.0, 0)
    ball = Ball(poseBall, 1.0, 100, RADIUS_BALL, behavierBall)
    for player in players:
        player.sensors.set_full_vision(full_vision)
        
    return Simulation(np.array(players), ball, shockable, full_vision)

  

def init_simulation(simulation):
    now = datetime.datetime.now()
    pygame.init()
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Robot soccer 2D environment")
    # icon = pygame.image.load(os.getcwd() + '/icon.PNG')
    # pygame.display.set_icon(icon)
    clock = pygame.time.Clock()

    environment = Environment(window)
    while (datetime.datetime.now() - now).seconds < 1:
        clock.tick(FREQUENCY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end_simulation()

        simulation.update()
        draw(simulation, window, environment)
    return window, clock, environment


def end_simulation():
    pygame.quit()


def get_state(agent, ball, players):
    # Define the state representation
    # Example: position of the agent and the ball
    return (agent.pose.position.x, agent.pose.position.y, ball.pose.position.x, ball.pose.position.y)

def get_reward(agent, ball):
    current_distance_to_ball = sqrt((agent.pose.position.x - ball.pose.position.x)**2 +
                                    (agent.pose.position.y - ball.pose.position.y)**2)

    if agent.last_distance_to_ball is not None:
        distance_change = agent.last_distance_to_ball - current_distance_to_ball
        reward = distance_change  # Reward is the change in distance
    else:
        reward = 0

    agent.last_distance_to_ball = current_distance_to_ball
    return reward

def action_to_command(action):
    # Define linear and angular speed values
    linear_speed = 1.5  # Adjust this value as needed
    angular_speed = 1.5  # Adjust this value as needed
    if action == 0:  # Up - Move forward
        return (linear_speed, 0)
    elif action == 1:  # Down - Move backward
        return (-linear_speed, 0)
    elif action == 2:  # Left - Turn left
        return (0,angular_speed)
    elif action == 3:  # Right - Turn right
        return (0, -angular_speed)
    
simulation = simulation2D(
        [QLearningAgent(4, Pose(300 * 0.01, 300 * 0.01, 0), 20, 20, 0.15),],
        False, 
        False)

def start_simulation(socketio, shared_data, lock):
   
    window, clock, environment = init_simulation(simulation)
    running = True
    while running:

        if (simulation.left_goal + simulation.right_goal) >= 5:
            running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if socketio:
            simulation_data = get_simulation_data()
            socketio.emit('simulation_data', json.dumps(simulation_data))
            socketio.sleep(0.1)  

        for agent in simulation.player: 
            current_state = get_state(agent, simulation.ball, simulation.player)
            action = agent.choose_action(current_state)
            command = action_to_command(action)
            agent.set_velocity(*command)

            simulation.update()

            new_state = get_state(agent, simulation.ball, simulation.player)
            reward = get_reward(agent, simulation.ball)
            agent.learn(current_state, action, reward, new_state, False)
            agent.update_exploration_rate()

        draw(simulation, window, environment)

    pygame.quit()
    end_simulation()



def get_simulation_data():
    return {'ball': {'x' : simulation.ball.pose.position.x, 'y': simulation.ball.pose.position.y},
            'player' : {'x':simulation.player[0].pose.position.x, 'y': simulation.player[0].pose.position.y}
            }