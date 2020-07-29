# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import numpy as np
import deepchem as dc
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from deepchem.feat.mol_graphs import ConvMol
from layers import GraphConv, GraphPool, GraphGather, TrimGraphOutput
# from deepchem.models.layers import GraphConv, GraphPool, GraphGather, TrimGraphOutput

# load the qm7 dataset
tasks, datasets, transformers = dc.molnet.load_qm7_from_mat(
    featurizer='GraphConv', move_mean=True)
train_dataset, valid_dataset, test_dataset = datasets

# set the hyperparameters and metrics for the model
batch_size = 64
num_epochs = 50
n_tasks = 1
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.mean_squared_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.r2_score, mode="regression")
]

def _make_shapes_consistent(output, labels):
    """Try to make inputs have the same shape by adding dimensions of size 1."""
    shape1 = output.shape
    shape2 = labels.shape
    len1 = len(shape1)
    len2 = len(shape2)
    if len1 == len2:
        return (output, labels)
    if isinstance(shape1, tf.TensorShape):
        shape1 = tuple(shape1.as_list())
    if isinstance(shape2, tf.TensorShape):
        shape2 = tuple(shape2.as_list())
    if len1 > len2 and all(i == 1 for i in shape1[len2:]):
        for i in range(len1 - len2):
            labels = tf.expand_dims(labels, -1)
        return (output, labels)
    if len2 > len1 and all(i == 1 for i in shape2[len1:]):
        for i in range(len2 - len1):
            output = tf.expand_dims(output, -1)
        return (output, labels)
    raise ValueError("Incompatible shapes for outputs and labels: %s versus %s" %
                   (str(shape1), str(shape2)))

def reshape_y_pred(y_true, y_pred):
    """
    Function for reshaping the predictions
    to eliminate negative probability outputs
    Parameters
    ----------
    arg1 | y_true: tf.float32
        True value of y
    arg2 | y_pred: tf.float32
        Predicted value of y
    Returns
    -------
    Array
        list
    """
    n_samples = len(y_true)
    return y_pred[:n_samples, :]

class QuantileLoss(dc.models.losses.Loss):
    """
    Custom Pinball loss function for training GCN
    Parameters
    ----------
    arg1 | tau: int
        Quantile value
    Returns
    -------
    Float
        tf.float32
    """
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def __call__(self, output, labels):
        output, labels = _make_shapes_consistent(output, labels)
        diff = tf.math.subtract(output, labels)
        return K.mean(K.maximum(self.tau * diff, (self.tau - 1) * diff), axis=-1)

def data_generator(dataset, batch_size=None, epochs=1, predict=False, pad_batches=True):
    """
    Data generator for training the deepchem GCN model
    a single hidden layer
    Parameters
    ----------
    arg1 | dataset: dc.data.DiskDataset
        Input dataset to the generator
    arg1 | batch_size: int
        Size of every mini batch
    arg3 | epochs: int
        Number of iterations/epochs
    arg4 | predict: bool
        Boolean for predicting on the inputs of the dataset
    arg5 | pad_batches: bool
        Boolean for padding the batches
    Returns
    -------
    Tuple
        (numpy.ndarray32, numpy.ndarray32, numpy.ndarray32)
    """
    for epoch in range(epochs):
        for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(64,
                                                                         pad_batches=pad_batches,
                                                                         deterministic=True)):
            # init the ConvMol obj to get the inputs
            multiConvMol = ConvMol.agglomerate_mols(X_b)

            # get the number of samples
            n_samples = np.array(X_b.shape[0])

            # prepare the inputs
            inputs = [multiConvMol.get_atom_features(),
                      multiConvMol.deg_slice,
                      np.array(multiConvMol.membership)]

            # add the degree adjacency list to the input
            for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                inputs.append(multiConvMol.get_deg_adjacency_lists()[i])

            outputs = [y_b]
            weights = [w_b]

            # yield on  X, y, w
            yield (inputs, outputs, weights)

def create_gcn_model():
    """
    Function for creating template GCN models with
    a single hidden layer
    Parameters
    ----------
    No arguments
    Returns
    -------
    Neural Network Model
        tf.keras.layers.Model
    """
    # setup the inputs for the model
    atom_features = layers.Input(shape=(75,))
    degree_slice = layers.Input(shape=(2,), dtype=tf.int32)
    membership = layers.Input(shape=tuple(), dtype=tf.int32)
    n_samples = layers.Input(shape=tuple(), dtype=tf.int32)

    deg_adjs = []
    for i in range(0, 10 + 1):
        deg_adj = layers.Input(shape=(i+1,), dtype=tf.int32)
        deg_adjs.append(deg_adj)

    # first GCN layer with 64 channels
    gc1 = GraphConv(64, activation_fn=tf.nn.relu)([atom_features,
                                                   degree_slice, membership] + deg_adjs)

    # Batch norm for the first GCN layer with 64 channels
    batch_norm1 = layers.BatchNormalization()(gc1)

    # pooling
    gp1 = GraphPool()([batch_norm1, degree_slice, membership] + deg_adjs)

    # second GCN layer with 64 channels
    gc2 = GraphConv(64, activation_fn=tf.nn.relu)(
        [gp1, degree_slice, membership] + deg_adjs)

    # Batch norm for the second GCN layer with 64 channels
    batch_norm2 = layers.BatchNormalization()(gc2)

    # pooling
    gp2 = GraphPool()([batch_norm2, degree_slice, membership] + deg_adjs)

    # Dense layer with 128 neurons
    dense = layers.Dense(128, activation=tf.nn.relu)(gp2)

    # Batch norm for the Dense layer with 128 neurons
    batch_norm3 = layers.BatchNormalization()(dense)

    # Gather the batch norm output and tie it with the graph
    readout = GraphGather(batch_size=batch_size,
                          activation_fn=tf.nn.tanh)([batch_norm3,
                                                     degree_slice, membership] + deg_adjs)

    # output layer
    out = layers.Dense(1, name='output')(readout)
    # out = TrimGraphOutput()([out, n_samples])

    # inputs = [atom_features, degree_slice, membership, n_samples] + deg_adjs
    inputs = [atom_features, degree_slice, membership] + deg_adjs
    outputs = [out]
    # outputs = [out, readout]

    # create the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    # return the model
    return model


# create the gcn model
gcn_model = create_gcn_model()

# setup the loss function
loss = dc.models.losses.L2Loss()
# loss = QuantileLoss(0.025)
# loss = QuantileLoss(0.5)
# loss = QuantileLoss(0.975)

# build the keras model of deepchem
model = dc.models.KerasModel(gcn_model, loss=loss)

# print model summary
model.model.summary()

# train the model and print the loss
losses = []
for i in range(num_epochs):
    loss = model.fit_generator(data_generator(train_dataset, epochs=1))
    print("Epoch %d loss: %f" % (i, loss))
    losses.append(loss)


print("Evaluating model")
preds = model.predict_on_generator(data_generator(test_dataset, predict=True))
preds = reshape_y_pred(test_dataset.y, preds)
mae_scores = metric[0].compute_metric(test_dataset.y, preds, test_dataset.w)
# mse_scores = metric[1].compute_metric(test_dataset.y, preds, test_dataset.w)
# r2_scores = metric[2].compute_metric(test_dataset.y, preds, test_dataset.w)
# print("Test MAE Score: %f" % mae_scores)
