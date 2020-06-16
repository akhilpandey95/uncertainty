# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import cupy as cp
import numpy as np
import tensorflow as tf

# feedforward network for compound target prediction
class SmilesLSTM(tf.keras.Model):
    """
    Class object for SmilesLSTM

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        tf.keras.Model

    """

    # function for preparing the X & Y for the dataset
    def __init__(self):
    """
    Build the Recurrent neural network model and compile it

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

        # add the first hidden layer with 64 neurons, relu activation
        self.dense1 = tf.keras.layers.Dense(512, activation='selu', input_dim=21)

        # add a 1D Convolutional layer with 32 kernels
        self.conv1 = tf.keras.layers.Conv1D(32, 4, strides=2, activation='selu', dropout=0.5)

        # add a max pooling layer to the 1D Convolutional layer
        self.maxpool1 = tf.keras.layers.MaxPool1D()

        # add an LSTM layer with 128 units
        self.lstm1 = tf.keras.layers.LSTM(128, dropout=0.1, return_sequences=True)

        # add an LSTM layer with 512 units
        self.lstm2 = tf.keras.layers.LSTM(512, dropout=0.1)

        # add the single output layer
        self.output = tf.keras.layers.Dense(bias_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0.000000001))

    # function for compiling the model
    @tf.function
    def call(self, x):
        """
        Compile the neural network model

        Parameters
        ----------
        arg1 | model: keras.model.Model
            A compiled keras neural network model to train
        arg2 | X: numpy.ndarray
            The training samples containing all the predictors

        Returns
        -------
        Neural Network Model
            tf.keras.Model

        """
        try:
            # activation on the first hidden layer with 64 neurons, relu activation
            x = self.dense1(x)

            # activation on a 1D Convolutional layer with 32 kernels
            x = self.conv1(x)

            # activation on a max pooling layer to the 1D Convolutional layer
            x = self.maxpool1(x)

            # activation on an LSTM layer with 128 units
            x = self.lstm1(x)

            # activation on an LSTM layer with 512 units
            x = self.lstm2(x)

            # return the output
            return self.output(x)
        except:
            return tf.keras.Model()

    # train the model
    @tf.function
    def train_step(self, features, labels):
        """
        Fit the neural network model

        Parameters
        ----------
        arg1 | model: keras.model.Model
            A compiled tensorflow neural network model to train
        arg2 | features: numpy.ndarray
            The training samples containing all the predictors
        arg3 | labels: numpy.ndarray
            The training samples containing values for the target variable
        
        Returns
        -------
        Array, Array
            numpy.ndarray, numpy.ndarray

        """
        self.train_loss_fn = tf.keras.metrics.Mean(name='train_loss')
        self.train_metrics_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='train_metrics')

        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            loss = loss_object(labels, predictions)

        # gather the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # backpropagate and optimize the gradients 
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # collect the losses
        self.train_losses = self.train_loss_fn(loss)

        # collect the metrics
        self.train_metrics = self.train_metrics_fn(labels, predictions)

        # return the losses and metrics
        return self.train_losses, self.train_metrics

    # evaluate the model
    @tf.function
    def test_step(self, features, labels):
        """
        Evaluate the neural network model

        Parameters
        ----------
        arg1 | model: keras.model.Model
            A trained tensorflow neural network model to test
        arg2 | features: numpy.ndarray
            The test samples containing all the predictors
        arg3 | labels: numpy.ndarray
            The test samples containing values for the target variable
        
        Returns
        -------
        Array, Array
            numpy.ndarray, numpy.ndarray

        """
        self.test_loss_fn = tf.keras.metrics.Mean(name='test_loss')
        self.test_metrics_fn = tf.keras.metrics.SparseCategoricalAccuracy(name='test_metrics')

        with tf.GradientTape() as tape:
            predictions = self.model(features, training=False)
            loss = loss_object(labels, predictions)

        # gather the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # backpropagate and optimize the gradients 
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # collect the losses
        self.test_losses = self.test_loss_fn(loss)

        # collect the metrics
        self.test_metrics = self.test_metrics_fn(labels, predictions)

        # return the losses and metrics
        return self.test_losses, self.test_metrics

    # function to put it all together
    def go(self, epochs=10):
        """
        Fit the neural network model

        Parameters
        ----------
        arg1 | model: int
            The number of iterations the neural network model has to run

        Returns
        -------
        Neural Network Model
            tf.keras.Model
        """
        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss_fn.reset_states()
            self.train_metrics_fn.reset_states()
            self.test_loss_fn.reset_states()
            self.test_metrics_fn.reset_states()

            for X_train, y_train in train_ds:
                self.train_step(X_train, y_train)

            for X_test, y_test in test_ds:
                self.test_step(X_test, y_test)

            # template for printing
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss_fn.result(),
                                  self.train_metrics_fn.result() * 100,
                                  self.test_loss_fn.result(),
                                  self.test_metrics_fn.result() * 100))


# function class for SmilesLSTM Bayesian network
class SmilesLSTMCell_Bayesian(tf.keras.layers.LSTMCell):
    """
    Class object for Bayesian recurrent network for SmilesLSTM architecture

    Parameters
    ----------
    No arguments

    Returns
    -------
    Neural Network Model
        tf.keras.Model

    """
    def __init__(self, num_units, prior, is_training, name, **kwargs):

        super(SmilesLSTM_Bayesian, self).__init__(num_units, **kwargs)

        self.w = None
        self.b = None
        self.prior = prior
        self.layer_name = name
        self.isTraining = is_training
        self.num_units = num_units
        self.kl_loss=None

    def call(self, inputs, state):

        if self.w is None:
            size = inputs.get_shape()[-1].value
            self.w, self.w_mean, self.w_sd = variationalPosterior((size+self.num_units, 4*self.num_units), self.layer_name+'_weights', self.prior, self.isTraining)
            self.b, self.b_mean, self.b_sd = variationalPosterior((4*self.num_units,1), self.layer_name+'_bias', self.prior, self.isTraining)

        cell, hidden = state
        concat_inputs_hidden = tf.concat([inputs, hidden], 1)
        concat_inputs_hidden = tf.nn.bias_add(tf.matmul(concat_inputs_hidden, self.w), tf.squeeze(self.b))
        # Gates: Input, New, Forget and Output
        i, j, f, o = tf.split(value=concat_inputs_hidden, num_or_size_splits=4, axis=1)

        # create the new states
        new_cell = (cell * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
        new_hidden = self._activation(new_cell) * tf.sigmoid(o)
        new_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_cell, new_hidden)

        # return the states
        return new_hidden, new_state

class SmilesLSTM:

    def __init__(self, training):

        self.LSTM_KL=0
        self.embedding_dim = 300  # the number of hidden units in each RNN
        self.keep_prob = 0.5
        self.batch_size = 512
        self.lstm_sizes = [128, 64]  # number hidden layer in each LSTM
        self.num_classes = 2
        self.max_sequence_length = 100
        self.prior=(0,1) #univariator prior
        self.isTraining=training


        with tf.variable_scope('rnn_i/o'):
            # use None for batch size and dynamic sequence length
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.embedding_dim])
            self.groundtruths = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        with tf.variable_scope('rnn_cell'):
            self.initial_state, self.final_lstm_outputs, self.final_state, self.cell = self.build_lstm_layers(self.lstm_sizes, self.inputs,self.keep_prob, self.batch_size)



            self.softmax_w, self.softmax_w_mean, self.softmax_w_std=  variationalPosterior((self.lstm_sizes[-1], self.num_classes), "softmax_w", self.prior, self.isTraining)
            self.softmax_b, self.softmax_b_mean, self.softmax_b_std = variationalPosterior((self.num_classes), "softmax_b", self.prior, self.isTraining)
            self.logits=tf.nn.xw_plus_b(self.final_lstm_outputs,  self.softmax_w,self.softmax_b)

        with tf.variable_scope('rnn_loss', reuse=tf.AUTO_REUSE):

            if (self.isTraining):
                self.KL=0.
                # use cross_entropy as class loss
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.groundtruths, logits=self.logits)
                self.KL=tf.add_n(tf.get_collection("KL_layers"), "KL")

            self.cost=(self.loss+self.KL)/self.batch_size  #the total cost need to divide by batch size
            self.optimizer = tf.train.AdamOptimizer(0.02).minimize(self.loss)

        #with tf.variable_scope('rnn_accuracy'):
            # self.accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.groundtruths, axis=1), predictions=self.prediction)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
        self.saver = tf.train.Saver()  # a saver is for saving or restoring your trained weight

        print("Completed creating the graph")

    def train(self, batch_x, batch_y, state):

        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = state
        # feed in input and groundtruth to get loss and update the weight via Adam optimizer
        loss, accuracy, final_state, _ = self.sess.run([self.loss, self.accuracy, self.final_state, self.optimizer], fd)

        return loss, accuracy, final_state

    def test(self, batch_x, batch_y, batch_size):

        """
         NEED TO RE-WRITE this function interface by adding the state
        :param batch_x:
        :param batch_y:
        :return

        """
        # restore the model

        # with tf.Session() as sess:
        #    model=model.restore();

        test_state = model.cell.zero_state(batch_size, tf.float32)
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = test_state
        prediction, accuracy = self.sess.run([self.prediction, self.accuracy], fd)

        return prediction, accuracy

    def save(self, e):
        self.saver.save(self.sess, 'model/rnn/rnn_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/rnn/rnn_%d.ckpt' % (e))

    def build_lstm_layers(self, lstm_sizes, inputs, keep_prob_, batch_size):
        """
        Create the LSTM layers
        inputs: array containing size of hidden layer for each lstm,
                input_embedding, for the shape batch_size, sequence_length, emddeding dimension [None, None, 384],
                None and None are to handle variable batch size and variable sequence length
                keep_prob for the dropout and batch_size

        outputs: initial state for the RNN (lstm) : tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)]
                 outputs of the RNN [Batch_size, sequence_length, last_hidden_layer_dim]
                 RNN cell: tensorflow implementation of the RNN cell
                 final state: tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)]

        """
        self.lstms=[]
        for i in range (0,len(lstm_sizes)):
            self.lstms.append(SmilesLSTMCell_Bayesian(lstm_sizes[i], self.prior, self.isTraining, 'lstm'+str(i)))

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell(self.lstms)
        # Getting an initial state of all zeros

        initial_state = cell.zero_state(batch_size, tf.float32)
        # perform dynamic unrolling of the network, for variable
        #lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, initial_state=initial_state)

        # we avoid dynamic RNN, as this produces while loop errors related to gradient checking
        if True:
            outputs = []
            state = initial_state
            with tf.variable_scope("RNN"):
                for time_step in range(self.max_sequence_length):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

        final_lstm_outputs = cell_output
        final_state = state
        #outputs=tf.reshape(tf.concat(1, outputs), [-1, self.embedding_dim])


        return initial_state, final_lstm_outputs, final_state, cell

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


