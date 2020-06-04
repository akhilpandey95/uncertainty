# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import cupy as cp
import numpy as np
import tensorflow as tf

# feedforward network for compound target prediction
class SmilesLSTM(Model):
    """
    Class object for SmilesLSTM

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        keras.model.Model

    """
    # function for preparing the X & Y for the dataset
    def __init__(self):
        """
        Build the Vanilla style neural network model and compile it

        Parameters
        ----------
        No arguments

        Returns
        -------
        Nothing
            None

        """
        # super class the keras model
        super(SmilesLSTM, self).__init__()

        # create the model
        self.model = Sequential()

        # add the first hidden layer with 64 neurons, relu activation
        self.model.add(tf.keras.layers.Dense(512, activation='selu', input_dim=21))

        # add a 1D Convolutional layer with 32 kernels
        self.model.add(tf.keras.layers.Conv1D(32, 4, strides=2, activation='selu', dropout=0.5))

        # add a max pooling layer to the 1D Convolutional layer
        self.model.add(tf.keras.layers.MaxPool1D())

        # add an LSTM layer with 128 units
        self.model.add(tf.keras.layers.LSTM(128, dropout=0.1, return_sequences=True))

        # add an LSTM layer with 512 units
        self.model.add(tf.keras.layers.LSTM(512, dropout=0.1))

        # add the single output layer
        self.model.add(tf.keras.layers.Dense())

        # add an L1 weight regularization
        self.model.add(tf.keras.regularizers.l1(l=0.0))

        # add an L2 weight regularization
        self.model.add(tf.keras.regularizers.l2(l=0.000000001))

        # use the rmsprop optimizer
        self.rms = tf.keras.optimizers.Adam(learning_rate=0.0001)

        # compile the model
        self.model.compile(optimizer=self.rms, loss='binary_crossentropy', metrics =['accuracy'])

    # function for training the neural network model
    def train(self, epochs, batch_size, X_train, X_test, Y_train, Y_test, stopping=True):
        """
        Fit the neural network model

        Parameters
        ----------
        arg1 | model: keras.model.Model
            A compiled keras neural network model to train
        arg2 | X_train: numpy.ndarray
            The training samples containing all the predictors
        arg3 | X_test: numpy.ndarray
            The test samples containing all the predictors
        arg4 | Y_train: numpy.ndarray
            The training samples containing values for the target variable
        arg5 | Y_test: numpy.ndarray
            The test samples containing values for the target variable
        arg6 | stopping: boolean
        A flag asserting if early stopping should or shouldn't be used for training

        Returns
        -------
        Neural Network Model
            keras.model.Model

        """
        try:
            if not stopping:
                # fit the model
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size)
            else:
                # prepare for early stopping
                early_stopping = keras.callbacks.EarlyStopping(monitor='binary_cross_entropy', min_delta=0,
                                                         patience=300, verbose=0, mode='auto',
                                                         baseline=None, restore_best_weights=False)
                # fit the model
                self.model.fit(X_train, Y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, callbacks=[early_stopping])

            # return the model
            return self.model
        except:
            return keras.models.Model()



