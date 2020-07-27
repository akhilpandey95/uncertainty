# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal

# class object for the gaussian layer
class GaussianLayer(tf.keras.layers.Layer):
    """
    Class object for gaussian layer
    Parameters
    ----------
    No arguments
    Returns
    -------
    Neural Network Layers
        tf.keras.layers.Layer
    """
    # init the gaussian

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GaussianLayer, self).__init__(**kwargs)

    # build the tf layer
    def build(self, input_shape):
        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=[int(input_shape[-1]),
                                               self.output_dim],
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=[int(input_shape[-1]),
                                               self.output_dim],
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                      shape=(self.output_dim, ),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                      shape=(self.output_dim, ),
                                      initializer=glorot_normal(),
                                      trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, input):
        output_mu = tf.matmul(input, self.kernel_1) + self.bias_1
        output_sig = tf.matmul(input, self.kernel_2) + self.bias_2
        output_sig_pos = tf.math.log(1 + tf.math.exp(output_sig)) + 1e-06

        # return the input
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]
