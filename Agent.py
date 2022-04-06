import numpy as np


class Agent:
    def __init__(self, observation_space, action_space, random=0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.random = random
        self.lastState = None
        self.lastAction = None


    def step(self, state: np.ndarray) -> int:
        self.lastState = state
        self.lastAction = 0
        return self.lastAction

    def update(self, new_state, reward) -> None:
        pass

