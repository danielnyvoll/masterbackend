import numpy as np

class QLearningModel:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with dimensions for state space + action space
        dimOfQ = np.concatenate((self.state_space, [self.action_space]))
        self.Q = np.ones(dimOfQ)

    def choose_action(self, s0, s1, s2):
        s2 = int(s2)
        Qvalues = self.Q[s0, s1, s2]
        # Pick the best action, tie-break randomly
        action = np.random.choice(np.flatnonzero(Qvalues == Qvalues.max()))
        return action

    def learn(self, alpha, s0, s1, s2, action, s_prime0, s_prime1, s_prime2, reward, done):
        s2 = int(s2)
        s_prime2 = int(s_prime2)
        
        # Calculate V_prime for end state; next state value is 0 if done
        V_prime = 0 if done else np.max(self.Q[s_prime0, s_prime1, s_prime2])
        
        # Update Q-value
        self.Q[s0, s1, s2, action] = \
            (1 - alpha) * self.Q[s0, s1, s2, action] + \
            alpha * ((1 - self.discount_factor) * reward + self.discount_factor * V_prime)

    def update_epsilon(self):
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
