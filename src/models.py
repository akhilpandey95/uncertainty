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


# Graph convolutional layer for building the GCN model
class GCNConv(tf.keras.layers.Layer):
        """
    Class object for graph convolutional layer

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        tf.keras.layers.Layer

    """
    def __init__(self,
                 units,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        # add the
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # super class the keras model
        super(GCNConv, self).__init__()

    # build the layer
    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        # gsize = input_shape[0][0]  # graph size
        fdim = input_shape[1][1]  # feature dim

        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                          shape=(fdim, self.units),
                                          initializer=self.kernel_initializer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(name="bias",
                                            shape=(self.units, ),
                                            initializer=self.bias_initializer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        # super class the keras model
        super(GCNConv, self).build(input_shape)

    # call the layer
    def call(self, inputs):
        """ GCN has two inputs : [An, X]
        """
        self.An = inputs[0]
        self.X = inputs[1]

        if isinstance(self.X, tf.SparseTensor):
            h = spdot(self.X, self.weight)
        else:
            h = dot(self.X, self.weight)
        output = spdot(self.An, h)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        return output

# Graph neural network (GNN) for learning graph representations
class GCN(Base):
    """
    Class object for graph neural network model

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        tf.keras.model.Model

    """
    # function for preparing the X & Y for the dataset
    def __init__(self, An, X, sizes, **kwargs):
        """
        Build the Graph neural network model and compile it

        Parameters
        ----------
        No arguments

        Returns
        -------
        Nothing
            None

        """
        # super class the keras model
        super(GCN, self).__init__()

        # obtain the feature matrix and the the graph
        self.An = An
        self.X = X
        self.layer_sizes = sizes
        self.shape = An.shape

        self.An_tf = sp_matrix_to_sp_tensor(self.An)
        self.X_tf = sp_matrix_to_sp_tensor(self.X)

        # add the convolutional layers
        self.layer1 = GCNConv(self.layer_sizes[0], activation='relu')
        self.layer2 = GCNConv(self.layer_sizes[1])

        # add the optmization function
        self.opt = tf.optimizers.Adam(learning_rate=self.lr)

    # function for training the neural network model
    def train(self, epochs, batch_size, X_train, Y_train, stopping=True):
        """
        Fit the neural network model

        Parameters
        ----------
        arg1 | model: tf.keras.model.Model
            A compiled tensorflow neural network model to train
        arg2 | epochs: numpy.int32
            Number of epochs needed to train the model
        arg3 | batch_size: numpy.int32
            Batch size of the model
        arg4 | X_train: numpy.ndarray
            The training samples containing all the predictors
        arg5 | Y_train: numpy.ndarray
            The training samples containing values for the target variable
        arg6 | stopping: boolean
        A flag asserting if early stopping should or shouldn't be used for training

        Returns
        -------
        Neural Network Model
            keras.model.Model

        """
        K = labels_train.max() + 1
        train_losses = []
        val_losses = []

        # use adam optimizer to optimize
        for it in range(FLAGS.epochs):
            tic = time()
            with tf.GradientTape() as tape:
                _loss = self.loss_fn(idx_train, np.eye(K)[labels_train])

            # optimize over weights
            grad_list = tape.gradient(_loss, self.var_list)
            grads_and_vars = zip(grad_list, self.var_list)
            self.opt.apply_gradients(grads_and_vars)

            # evaluate on the training
            train_loss, train_acc = self.evaluate(idx_train, labels_train, training=True)
            train_losses.append(train_loss)
            val_loss, val_acc = self.evaluate(idx_val, labels_val, training=False)
            val_losses.append(val_loss)
            toc = time()
            if self.verbose:
                print("iter:{:03d}".format(it),
                      "train_loss:{:.4f}".format(train_loss),
                      "train_acc:{:.4f}".format(train_acc),
                      "val_loss:{:.4f}".format(val_loss),
                      "val_acc:{:.4f}".format(val_acc),
                      "time:{:.4f}".format(toc - tic))

        # return the losses from training
        return train_losses

    # function for evaluating the model and reporting stats
    def evaluate(self, idx, true_labels, training):
        """
        Evaluate the graph neural network model

        Parameters
        ----------
        arg1 | model: keras.model.Model
            A trained TF graph neural network model
        arg2 | option: str
            A flag asserting whether to evaluate either train or test samples
        arg3 | **data: variable function arguments
            The variable argument used for pulling the training or test data

        Returns
        -------
        Tuple
            numpy.ndarray, numpy.ndarray

        """
        K = true_labels.max() + 1

        # compute the losses
        _loss = self.loss_fn(idx, np.eye(K)[true_labels], training=training).numpy()

        # predict the probabilities
        _pred_logits = tf.gather(self.h2, idx)

        # assign the label based on the probabilities
        _pred_labels = tf.argmax(_pred_logits, axis=1).numpy()

        # calculate the accuracy score[evaluation metric]
        _acc = accuracy_score(_pred_labels, true_labels)

        # return the losses and metrics
        return _loss, _acc


