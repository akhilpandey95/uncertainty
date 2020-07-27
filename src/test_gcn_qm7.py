# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import numpy as np
import deepchem as dc
import tensorflow as tf
from tensorflow.keras import layers
from deepchem.feat.mol_graphs import ConvMol
from layers import GraphConv, GraphPool, GraphGather

# load the qm7 dataset
tasks, datasets, transformers = dc.molnet.load_qm7_from_mat(
    featurizer='GraphConv', move_mean=True)
train_dataset, valid_dataset, test_dataset = datasets

# set the hyperparameters and metrics for the model
batch_size = 64
num_epochs = 5
n_tasks = 1
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]


def data_generator(dataset, epochs=1, predict=False, pad_batches=True):
    """
    Data generator for training the deepchem GCN model
    a single hidden layer
    Parameters
    ----------
    No arguments
    arg1 | dataset: dc.data.DiskDataset
        Input dataset to the generator
    arg2 | epochs: int
        Number of iterations/epochs
    arg3 | predict: bool
        Boolean for predicting on the inputs of the dataset
    arg4 | pad_batches: bool
        Boolean for padding the batches
    Returns
    -------
    Tuple
        (numpy.ndarray32, numpy.ndarray32, numpy.ndarray32)
    """
    for epoch in range(epochs):
        for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size,
                                                                         pad_batches=pad_batches,
                                                                         deterministic=True)):
            # init the ConvMol obj to get the inputs
            multiConvMol = ConvMol.agglomerate_mols(X_b)

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

    inputs = [atom_features, degree_slice, membership] + deg_adjs

    # create the model
    model = tf.keras.Model(inputs=[inputs], outputs=[out])

    # return the model
    return model


# create the gcn model
gcn_model = create_gcn_model()

# setup the loss function
loss = dc.models.losses.L1Loss()

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
