import numpy as np

from Agent import Agent

PATH = 'Saves/qTableSave'

class QTable:
    def __init__(self, observation_space, action_space, chunks=10):
        shape = ((chunks,) * len(observation_space)) + (action_space,)
        self.qTable = np.random.randn(*shape)
        self.spaces = np.array([np.linspace(start, end, chunks-1) for start, end in observation_space])

    def __getitem__(self, item):
        index = [np.argmax(spaces > value) for spaces, value in zip(self.spaces, item)]
        return self.qTable[(*index,)]

    def get(self, state, action):
        return self[state][action]

    def set(self, state, action, value):
        self[state][action] = value

    def save(self, path=None):
        name = PATH if path is None else "Saves/" + path
        np.savez(name, qTable=self.qTable, spaces=self.spaces)
        print(f"Agent saved to {name}.npz")

    def load(self, path=None):
        name = PATH if path is None else "Saves/" + path
        npzfile = np.load(name + ".npz")
        self.qTable, self.spaces = npzfile['qTable'], npzfile['spaces']
        assert isinstance(self.qTable, np.ndarray), f"loaded qTable has wrong Type {type(self.qTable)}"
        assert isinstance(self.spaces, np.ndarray), f"loaded spaces has wrong Type {type(self.spaces)}"

class QTableAgent(Agent):
    def __init__(self, observation_space, action_space, chunks=10, alpha=.1, gamma=.9, beta=1, epsilon=.0):
        super().__init__(observation_space, action_space, epsilon=epsilon)
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.qTable = QTable(observation_space, action_space, chunks)

    def step(self, state: np.ndarray) -> int:
        self.lastState = state
        if np.random.randint(100) / 100 >= self.epsilon:
            action_values = self.qTable[state]
            self.lastAction = int(np.argmax(action_values))
        else:
            self.lastAction = np.random.randint(self.action_space)

        return self.lastAction

    def update(self, newState, reward):
        if self.lastState is not None and self.lastAction is not None:
            Q_t = self.qTable[self.lastState][self.lastAction]
            new_value = (1 - self.alpha) * Q_t + self.alpha * (reward + self.gamma * np.max(self.qTable[newState]))
            self.qTable.set(self.lastState, self.lastAction, new_value)
            self.alpha *= self.beta

    def save(self):
        self.qTable.save()

    def load(self):
        self.qTable.load()




if __name__ == '__main__':
    start, end = -2, 2
    observation_space = [[start, end]] * 4
    action_space = 4
    agent = QTableAgent(observation_space, action_space, epsilon=.5)

    obs = np.random.randn(8)
    print(obs)
    action = agent.step(obs)
    print(action)
    obs = np.random.randn(8)
    print(obs)
    agent.update(obs, 1)