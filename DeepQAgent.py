import random
import time

import numpy as np
from collections import deque

from Agent import Agent
from Helper import Log

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense

log = Log()


class DeepQAgent(Agent):
    def __init__(self, observation_space, action_space, alpha=.1, gamma=.999, gamma_decay=3e-4, epsilon=0, epsilon_min=0, theta=1,
                 learning_rate=0.001, buffer_size=1000, batch_size=16, transfer_every=250):
        super().__init__(observation_space, action_space, epsilon, epsilon_min, theta)

        self.lastPrediction = None
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_decay = 1 - gamma_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.transfer_every = transfer_every
        self.lr = learning_rate

        self.operationNetwork = self.get_model()
        self.trainNetwork = self.get_model()

        self.memory = Memory(self.buffer_size, self.batch_size, self.observation_space, self.action_space)

        self.train_pointer = 0
        self.version = 0

        self.losses = []

        self.transfer_weights()

        self.print_prams()

    def step(self, state: np.ndarray) -> int:
        self.lastState = state

        assert isinstance(self.lastState, np.ndarray), f"lastState is not type ndarray instate {type(self.lastState)}"
        assert self.lastState.shape == (self.observation_space,), \
            f"lastState does not have the right shape got {self.lastState.shape} needed {(self.observation_space,)}"

        self.lastPrediction = self.operationNetwork.predict(np.array([self.lastState]))
        if np.random.uniform() >= self.epsilon:
            self.lastAction = int(np.argmax(self.lastPrediction))
        else:
            self.lastAction = random.randint(0, self.action_space-1)
            self.epsilon = np.max([self.epsilon * (1 - self.theta), self.epsilon_min])

        assert isinstance(self.lastAction, int), f"lastAction is not int ({type(self.lastAction)})"
        assert self.lastAction in [0, 1], f"lastAction not posible ({self.lastAction})"
        return self.lastAction

    def update(self, new_state, reward, done):
        assert reward == 1 or reward == -1, f"reward is {reward}!"
        if self.lastState is not None and self.lastAction is not None:

            self.memory.append(self.lastState, self.lastAction, reward, not done, new_state)
            if done:
                self.memory.append(self.lastState, self.lastAction, reward, not done, new_state)

            self.train_pointer += 1

            self.train()

    def train(self):
        if self.train_pointer <= self.batch_size:
            return

        x_train, y_train = self.create_batch()

        assert isinstance(x_train, np.ndarray), f"x_batch has the wrong type ({type(x_train)})"
        assert isinstance(y_train, np.ndarray), f"y_batch has the wrong type ({type(y_train)})"

        history = self.trainNetwork.fit(x=x_train, y=y_train, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=True)

        self.losses.append(history.history['loss'][0])
        assert len(history.history['loss']) == 1, f"{history.history['loss']}"

        if self.train_pointer % self.transfer_every == 0:
            self.transfer_weights()

    def create_batch(self):
        x_train, actions, rewards, dones, new_states = self.memory.batch()
        y_train = self.operationNetwork.predict(x_train)
        y_train[np.arange(self.batch_size), actions] = rewards + dones * (1 - self.gamma) * np.max(self.operationNetwork.predict(new_states), axis=1)

        self.gamma *= self.gamma_decay

        assert x_train.shape == (self.batch_size, self.observation_space), f"x_train has shape {x_train.shape}"
        assert y_train.shape == (self.batch_size, self.action_space), f"y_train has shape {y_train.shape}"

        log(x_train, y_train)

        return x_train, y_train


    def create_trainingsset(self):
        logs = []

        start = time.time()
        batch = random.sample(self.memory, self.batch_size)
        x, y = [], []
        logs.append(f"sample: {time.time() - start}")

        start_iteration = time.time()
        for state, action, reward, new_state in batch:

            start = time.time()
            x.append(state[0])
            logs.append(f"append state: {time.time() - start}")

            start = time.time()
            target = self.operationNetwork.predict(state)[0]
            logs.append(f"predict target: {time.time() - start}")

            start = time.time()
            assert target.shape == (2,), f"y has the wrong shape ({target.shape})"
            logs.append(f"assert: {time.time() - start}")

            start = time.time()
            target[action] = reward if reward == -1 else reward + self.gamma * np.max(self.operationNetwork.predict(new_state))
            logs.append(f"set new target: {time.time() - start}")

            start = time.time()
            y.append(target)
            logs.append(f"append target: {time.time() - start}")

        logs.append(f"batch: {time.time() - start_iteration}")
        log.log(logs)

        return np.array(x), np.array(y)


    def get_model(self):
        # Create a simple model.
        inputs = keras.Input(shape=(self.observation_space,))
        # Norm = BatchNormalization()(inputs)

        # Dense1 = Dense(32, activation='tanh')(inputs)
        # Norm2 = BatchNormalization()(Dense1)

        Dense2 = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
        # Norm3 = BatchNormalization()(Dense2)

        Dense3 = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(Dense2)
        # Norm4 = BatchNormalization()(Dense3)

        Dense4 = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(Dense3)
        # Norm5 = BatchNormalization()(Dense4)

        outputs = Dense(2, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal())(Dense4)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        model.trainable = True
        return model

    def transfer_weights(self):
        self.operationNetwork.set_weights(self.trainNetwork.get_weights())
        self.version += 1

    def save(self, name):
        path = "Saves/" + name + ".h5"
        self.operationNetwork.save(path)

    def load(self, name):
        path = "Saves/" + name + ".h5"
        self.operationNetwork = tf.keras.load_model(path)
        self.transfer_weights()

    def print_prams(self):
        self.operationNetwork.summary()
        print()
        print(f"epsilon: {self.epsilon}\n"
              f"lr: {self.lr}\n"
              f"epsilon: {self.epsilon}\n"
              f"theta: {self.theta}\n"
              f"gamma: {self.gamma}\n"
              f"gamma_decay: {self.gamma_decay}\n"
              f"batch_size: {self.batch_size}\n"
              f"buffer: {self.buffer_size}\n"
              f"transfer_every: {self.transfer_every}\n\n")


class Memory:
    def __init__(self, buffer_size, batch_size, observation_space, action_space):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.pointer = 0
        self.size = 0

        self.states = np.zeros(shape=(buffer_size, observation_space))
        self.actions = np.zeros(shape=(buffer_size, ), dtype=np.int32)
        self.rewards = np.zeros(shape=(buffer_size,))
        self.new_states = np.zeros(shape=(buffer_size, observation_space))
        self.dones = np.zeros(shape=(buffer_size,))

    def append(self, state, action, reward, done, new_state):

        assert state.shape == (self.observation_space, ), f"appended state hase wrong shape ({state.shape})"
        assert new_state.shape == (self.observation_space,), f"appended new_state hase wrong shape ({new_state.shape})"
        assert isinstance(action, int), f"appended action hase wrong type ({type(action)})"
        # assert reward == 1 or reward == -1, f"appended reward is not in [1, -1] ({reward})"
        assert isinstance(done, bool) or isinstance(done, np.bool_), f"appended done hase wrong type ({type(done)})"

        self.size = min(self.size + 1, self.buffer_size)
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.new_states[self.pointer] = new_state
        self.dones[self.pointer] = done

    def batch(self):
        assert self.size >= self.batch_size, f"not enough datapoints in Memory have {self.size}"

        sample = np.random.randint(0, self.size, self.batch_size)

        return np.array(self.states[sample]), self.actions[sample], self.rewards[sample], self.dones[sample], np.array(self.new_states[sample])


if __name__ == '__main__':
    memory = Memory(500, 16, 4, 2)

    rews = np.array([-1, 1])
    do = np.array([True, False])
    for x in range(500):
        state, action, reward,done, new_state = np.random.randn(4), np.random.randint(0, 1), np.random.choice(rews, 1, replace=False)[0], np.random.choice(do, 1, replace=False)[0], np.random.randn(4)
        memory.append(state, action, reward, done, new_state)

    state, action, reward, done, new_state = memory.batch()
    print(state.shape, action.shape, reward.shape, new_state.shape)