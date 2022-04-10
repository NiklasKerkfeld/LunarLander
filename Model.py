import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


class DeepQModel:
    def __init__(self, input_size, output_size, batch_size, learning_rate):
        self.batch_size = batch_size

        self.target_model = self._create_model(input_size, output_size, learning_rate)
        self.train_model = self._create_model(input_size, output_size, learning_rate)
        self.version = 0

        self.train_counter = 0

        self.transfer_weights()

    def predict(self, state):
        return self.target_model.predict(state)

    def train(self, x_train, y_train):
        history = self.train_model.fit(x_train, y_train, batch_size=self.batch_size, verbose=0)
        self.train_counter += 1

        return history.history['loss'][0]

    def transfer_weights(self):
        self.target_model.set_weights(self.train_model.get_weights())
        self.version += 1

    def _create_model(self, input_size, output_size, learning_rate):
        """
         create a simple model
         :return: model
         """
        inputs = Input(shape=(input_size,))

        dense2 = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)

        dense3 = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(dense2)

        dense4 = Dense(8, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(dense3)

        outputs = Dense(output_size, activation='linear', kernel_initializer=tf.keras.initializers.HeNormal())(
            dense4)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        model.trainable = True

        return model

    def save(self, name):
        self.target_model.save("Saves/" + name + "_a" + ".h5")

    def load(self, name):
        self.train_model = tf.keras.load_model("Saves/" + name + ".h5")
        self.transfer_weights()
