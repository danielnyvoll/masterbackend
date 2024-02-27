import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DeepQLearningModel:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self._build_model()

    def _build_model(self):
        """Builds a simple neural network."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Returns actions by epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            print("||||||||||||||||||||||||||||||||||||||||||||||")
            return random.randrange(self.action_size)
        print("----------------------------------------")
        act_values = self.model.predict(state) 
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """Trains the model on a minibatch of experiences from the memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Loads a saved model."""
        self.model.load_weights(name)

    def save(self, name):
        """Saves the model."""
        self.model.save_weights(name)
