# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import numpy as np
import tensorflow as tf

# custom gaussian loss function
class GaussLoss(tf.keras.losses.Loss):
    """
    Class object for gaussian loss
    Parameters
    ----------
    arg1 | sigma: float32
        Value of sigma
    Returns
    -------
    Neural Network Loss Function
        tf.keras.losses.Loss
    """

    def __init__(self, sigma, name='GaussLoss'):
        super().__init__(name=name)
        self.sigma = sigma

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(0.5*tf.math.log(self.sigma) +
                                   0.5*tf.math.divide(tf.math.square(y_true - y_pred), self.sigma)) + 1e-6


def neg_log_likelihood_loss(y_true, y_pred):
    """
    Kerass loss Function for calculating
    negative log likelihood loss
    Parameters
    ----------
    arg1 | y_true: tf.float32
        True value of y
    arg2 | y_pred: tf.float32
        Predicted value of y
    Returns
    -------
    Float
        tf.keras.losses.Loss
    """
    sep = y_pred.shape[1] // 2
    mu, logvar = y_pred[:, :sep], y_pred[:, sep:]
    return K.sum(0.5*(logvar+np.log(2*np.pi)+K.square((y_true-mu)/K.exp(0.5*logvar))), axis=-1)
