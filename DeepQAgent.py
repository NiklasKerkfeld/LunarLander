import numpy as np

from Agent import Agent

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Dense

class DeepQAgent(Agent):
    def __init__(self, observation_space, action_space, alpha=.2, gamma=.9, random=0, buffer_size=1000, batch_size=32):
        super().__init__(observation_space, action_space, random)

        self.lastPrediction = None
        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.Q = self.get_model()
        self.x_train = np.zeros((buffer_size, observation_space))
        self.y_train = np.zeros((buffer_size, action_space))
        self.train_pointer = 0

    def step(self, state: np.ndarray) -> int:
        self.lastState = np.array([state])

        assert isinstance(self.lastState, np.ndarray), f"lastState is not type ndarray instate {type(self.lastState)}"
        assert self.lastState.shape == (1, self.observation_space, ), \
            f"lastState does not have the right shape got {self.lastState.shape} needed {(self.observation_space, )}"

        self.lastPrediction = self.Q.predict(self.lastState)
        if np.random.randint(100) / 100 >= self.random:
            self.lastAction = np.argmax(self.lastPrediction)
        else:
            self.lastAction = np.random.randint(self.action_space)

        return self.lastAction

    def update(self, new_state, reward):
        if self.lastState is not None and self.lastAction is not None:
            target = self.lastPrediction[0]
            assert target.shape == (2, ), f"target got the wrong shape expected (2,) got {target.shape} instate."
            target[self.lastAction] = (1 - self.alpha) * target[self.lastAction] + \
                                      self.alpha * (reward + self.gamma * np.max(self.Q.predict(np.array([new_state]))))
            self.x_train[self.train_pointer % self.buffer_size] = self.lastState
            self.y_train[self.train_pointer % self.buffer_size] = target

            self.train_pointer += 1

            self.train()


    def train(self):
        if self.train_pointer <= self.batch_size:
            return

        sample = np.random.randint(0, min(self.train_pointer, self.buffer_size), self.batch_size)
        x_batch = self.x_train[sample]
        y_batch = self.y_train[sample]

        assert isinstance(x_batch, np.ndarray), f"x_batch has the wrong type ({type(x_batch)})"
        assert isinstance(y_batch, np.ndarray), f"y_batch has the wrong type ({type(y_batch)})"

        self.Q.train_on_batch(x=x_batch, y=y_batch)

    def get_model(self):
        # Create a simple model.
        inputs = keras.Input(shape=(self.observation_space,))
        Norm = BatchNormalization()(inputs)
        Dense1 = Dense(8, activation='relu')(Norm)
        Dense2 = Dense(4, activation='relu')(Dense1)
        outputs = Dense(2, activation='softmax')(Dense2)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        model.trainable = True
        return model

