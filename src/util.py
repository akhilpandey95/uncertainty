# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import numpy as np
import tensorflow as tf


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
