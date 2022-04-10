import random
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

from Agent import Agent
from Memory import Memory
from Model import DeepQModel


class DoubleDeepQAgent(Agent):
    def __init__(self, observation_space,
                 action_space,
                 epsilon=1,
                 epsilon_min=1e-2,
                 theta=1-3e-4,
                 gamma=.9,
                 gamma_min=1e-3,
                 gamma_decay=1-1e-4,
                 buffer_size=1000,
                 batch_size=32,
                 learning_rate=1e-3,
                 transfer_every=300):
        """
        Agent for gym environment using Double Deep Q Learning
        :param observation_space: size of observations_space
        :param action_space: number of possible actions
        :param epsilon: rate of random actions
        :param epsilon_min: minimum rate of random actions
        :param theta: decay for random rate
        :param gamma: multiplyer for estimate reward (1 - gamma)
        :param gamma_min: minimum of gamma
        :param gamma_decay: gamma decay
        :param buffer_size: size of the Experience Replay buffer
        :param batch_size: size of the trainingsbatch
        :param learning_rate: learning rate for the models
        """
        super().__init__(observation_space, action_space, epsilon, epsilon_min, theta)

        self.gamma = gamma
        self.gamma_min = gamma_min
        self.gamma_decay = gamma_decay

        self.buffer_size = buffer_size

        self.batch_size = batch_size
        self.lr = learning_rate

        self.memory = Memory(buffer_size, batch_size, observation_space, action_space)

        self.model_a = DeepQModel(observation_space, action_space, batch_size, learning_rate)
        self.model_b = DeepQModel(observation_space, action_space, batch_size, learning_rate)

        self.train_switch = True
        self.transfer_every = transfer_every
        self.train_pointer = 0

        self.losses = []

    @property
    def version(self):
        return self.model_a.version

    def step(self, state: np.ndarray) -> int:
        """
        model predicts "good" action for a given state
        :param state: state to predict on
        :return: number from action_space
        """
        self.lastState = state

        assert isinstance(self.lastState, np.ndarray), f"lastState is not type ndarray instate {type(self.lastState)}"
        assert self.lastState.shape == (self.observation_space,), \
            f"lastState does not have the right shape got {self.lastState.shape} needed {(self.observation_space,)}"

        if np.random.uniform() < self.epsilon:
            self.lastAction = random.randint(0, self.action_space-1)
            self.epsilon = np.max([self.epsilon * self.theta, self.epsilon_min])

        else:
            if np.random.uniform() < 0.5:
                self.lastAction = int(np.argmax(self.model_a.predict(np.array([state]))))
            else:
                self.lastAction = int(np.argmax(self.model_b.predict(np.array([state]))))

        assert isinstance(self.lastAction, int), f"lastAction is not int ({type(self.lastAction)})"
        assert self.lastAction in [0, 1], f"lastAction not posible ({self.lastAction})"
        return self.lastAction

    def update(self, new_state, reward, done) -> None:
        """
        add experiance to Experience Replay buffer and train on a batch
        :param new_state: new state after last action
        :param reward: reward for the last action
        :param done: True if simulation is over
        :return: None
        """
        self.memory.append(self.lastState, self.lastAction, reward, not done, new_state)

        self.train()

    def train(self):
        """
        train a model with a sample from the Experience Replay buffer
        :return: None
        """
        if self.memory.ready:
            train_model, eval_model = (self.model_a, self.model_b) if self.train_switch else (self.model_b, self.model_a)

            x_train, actions, rewards, dones, new_states = self.memory.batch()
            target = train_model.predict(x_train)
            target[np.arange(self.batch_size), actions] = rewards + dones * (1 - self.gamma) * np.max(eval_model.predict(new_states))

            loss = train_model.train(x_train, target)
            self.losses.append(loss)

            self.gamma = max(self.gamma * self.gamma_decay, self.gamma_min)
            self.train_switch = not self.train_switch

            if train_model.train_counter % self.transfer_every == 0:
                train_model.transfer_weights()

    def save(self, name):
        self.model_a.save(name)
        self.model_b.save(name)

    def load(self, name):
        self.model_a.load(name + "_a")
        self.model_b.load(name + "_b")
