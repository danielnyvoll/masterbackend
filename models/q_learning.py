import numpy as np
import os
import json

class QLearningModel:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, discretization_steps=10):
        # Dimensions for discretization
        self.discretization = (240 // discretization_steps, 140 // discretization_steps)  # Given field dimensions and desired discretization steps
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with dimensions for the discretized state space + action space
        self.Q = np.random.uniform(low=-1, high=1, size=(self.discretization[0], self.discretization[1], self.discretization[0], self.discretization[1], 2, self.action_space))

    def discretize_state(self, player_pos, ball_pos, isGoal):
        # Convert continuous positions to discretized grid positions
        disc_player_x = np.digitize(player_pos['x'], bins=np.linspace(-120, 120, self.discretization[0])) - 1
        disc_player_y = np.digitize(player_pos['y'], bins=np.linspace(-70, 70, self.discretization[1])) - 1
        disc_ball_x = np.digitize(ball_pos['x'], bins=np.linspace(-120, 120, self.discretization[0])) - 1
        disc_ball_y = np.digitize(ball_pos['y'], bins=np.linspace(-70, 70, self.discretization[1])) - 1
        isGoal = 1 if isGoal else 0
        return (disc_player_x, disc_player_y, disc_ball_x, disc_ball_y, isGoal)

    def choose_action(self, state):
        if state.shape == (1, 4):  # If 'state' is passed as a 2D array with shape (1, 4)
            state = state.flatten()
        isGoal = False
        player_pos = {'x': state[0], 'y': state[1]}
        ball_pos = {'x': state[2], 'y': state[3]}

    # Correctly call discretize_state with the dictionaries and isGoal
        discretized_state = self.discretize_state(player_pos, ball_pos, isGoal)

    # Your existing logic for choosing an action based on the discretized state
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.Q[discretized_state])

    def learn(self, state, action, reward, next_state, done):
        if not isinstance(state, np.ndarray) or state.shape != (4,):
            raise ValueError(f"Expected 'state' to be a flat NumPy array with shape (4,), got {state} with shape {state.shape}")
    
        player_pos = {'x': state[0], 'y': state[1]}
        ball_pos = {'x': state[2], 'y': state[3]}

        isGoal = done

        discretized_state = self.discretize_state(player_pos, ball_pos, isGoal)
        discretized_next_state = self.discretize_state({'x': next_state[0], 'y': next_state[1]}, {'x': next_state[2], 'y': next_state[3]}, isGoal)
        
        # Update Q-value with the discretized states
        max_future_q = np.max(self.Q[discretized_next_state])
        current_q = self.Q[discretized_state + (action,)]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q * (not done))
        self.Q[discretized_state + (action,)] = new_q

        # Epsilon update
        self.update_epsilon()


    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        # Flatten Q-table and save as JSON
        flattened_q_table = self.Q.flatten().tolist()
        with open(filename, 'w') as f:
            json.dump(flattened_q_table, f)

    def load(self, filename):
        # Load and reshape Q-table from JSON
        with open(filename, 'r') as f:
            flattened_q_table = np.array(json.load(f))
            self.Q = flattened_q_table.reshape(self.Q.shape)
