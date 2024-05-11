from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_shape, action_size, model_file='dqn_model.keras', replay_memory_size=50000, minibatch_size=32):
        self.state_shape = state_shape
        self.action_size = action_size
        self.model_file = model_file
        self.epsilon = 1.0  # Starting value of epsilon
        self.epsilon_min = 0.01  # Minimum value of epsilon
        self.epsilon_decay = 0.9995  # Decay multiplier for epsilon
        self.model = self.load_or_create_model()
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.minibatch_size = minibatch_size

    def load_or_create_model(self):
        if os.path.exists(self.model_file):
            print("Loading existing model.")
            return load_model(self.model_file)
        else:
            print("No model found, initializing a new one.")
            return self.create_model()

    def create_model(self):
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', input_shape=self.state_shape),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def save_model(self):
        self.model.save(self.model_file)


    def add_experience(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_memory) < self.minibatch_size:
            return
        minibatch = random.sample(self.replay_memory, self.minibatch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + 0.99 * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if len(self.replay_memory) > 100:
            self.replay_memory.clear()
