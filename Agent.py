import numpy as np


class Agent:
    def __init__(self, observation_space, action_space, epsilon=0, epsilon_min=0, theta=1):
        self.observation_space = observation_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.theta = theta
        self.lastState = None
        self.lastAction = None


    def step(self, state: np.ndarray) -> int:
        self.lastState = state
        self.lastAction = 0
        return self.lastAction

    def update(self, new_state, reward, done) -> None:
        pass

    def save(self, name):
        pass

    def load(self, name):
        pass

