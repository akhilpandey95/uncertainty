# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal

import collections
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy
from deepchem.data import NumpyDataset, pad_features
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization

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


class TrimGraphOutput(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(TrimGraphOutput, self).__init__(**kwargs)

    def call(self, inputs):
        n_samples = tf.squeeze(inputs[1])
        return inputs[0][0:n_samples]

class GraphConv(tf.keras.layers.Layer):

  def __init__(self,
               out_channel,
               min_deg=0,
               max_deg=10,
               activation_fn=None,
               **kwargs):
    super(GraphConv, self).__init__(**kwargs)
    self.out_channel = out_channel
    self.min_degree = min_deg
    self.max_degree = max_deg
    self.activation_fn = activation_fn

  def build(self, input_shape):
    # Generate the nb_affine weights and biases
    num_deg = 2 * self.max_degree + (1 - self.min_degree)
    self.W_list = [
        self.add_weight(
            name='kernel',
            shape=(int(input_shape[0][-1]), self.out_channel),
            initializer='glorot_uniform',
            trainable=True) for k in range(num_deg)
    ]
    self.b_list = [
        self.add_weight(
            name='bias',
            shape=(self.out_channel,),
            initializer='zeros',
            trainable=True) for k in range(num_deg)
    ]
    self.built = True

  def call(self, inputs):

    # Extract atom_features
    atom_features = inputs[0]

    # Extract graph topology
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    W = iter(self.W_list)
    b = iter(self.b_list)

    # Sum all neighbors using adjacency matrix
    deg_summed = self.sum_neigh(atom_features, deg_adj_lists)

    # Get collection of modified atom features
    new_rel_atoms_collection = (self.max_degree + 1 - self.min_degree) * [None]

    for deg in range(1, self.max_degree + 1):
      # Obtain relevant atoms for this degree
      rel_atoms = deg_summed[deg - 1]

      # Get self atoms
      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Apply hidden affine to relevant atoms and append
      rel_out = tf.matmul(rel_atoms, next(W)) + next(b)
      self_out = tf.matmul(self_atoms, next(W)) + next(b)
      out = rel_out + self_out

      new_rel_atoms_collection[deg - self.min_degree] = out

    # Determine the min_deg=0 case
    if self.min_degree == 0:
      deg = 0

      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Only use the self layer
      out = tf.matmul(self_atoms, next(W)) + next(b)

      new_rel_atoms_collection[deg - self.min_degree] = out

    # Combine all atoms back into the list
    atom_features = tf.concat(axis=0, values=new_rel_atoms_collection)

    if self.activation_fn is not None:
      atom_features = self.activation_fn(atom_features)

    return atom_features

  def sum_neigh(self, atoms, deg_adj_lists):
    """Store the summed atoms by degree"""
    deg_summed = self.max_degree * [None]

    # Tensorflow correctly processes empty lists when using concat
    for deg in range(1, self.max_degree + 1):
      gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
      # Sum along neighbors as well as self, and store
      summed_atoms = tf.reduce_sum(gathered_atoms, 1)
      deg_summed[deg - 1] = summed_atoms

    return deg_summed


class GraphPool(tf.keras.layers.Layer):

  def __init__(self, min_degree=0, max_degree=10, **kwargs):
    super(GraphPool, self).__init__(**kwargs)
    self.min_degree = min_degree
    self.max_degree = max_degree

  def call(self, inputs):
    atom_features = inputs[0]
    deg_slice = inputs[1]
    deg_adj_lists = inputs[3:]

    # Perform the mol gather
    # atom_features = graph_pool(atom_features, deg_adj_lists, deg_slice,
    #                            self.max_degree, self.min_degree)

    deg_maxed = (self.max_degree + 1 - self.min_degree) * [None]

    # Tensorflow correctly processes empty lists when using concat

    for deg in range(1, self.max_degree + 1):
      # Get self atoms
      begin = tf.stack([deg_slice[deg - self.min_degree, 0], 0])
      size = tf.stack([deg_slice[deg - self.min_degree, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)

      # Expand dims
      self_atoms = tf.expand_dims(self_atoms, 1)

      # always deg-1 for deg_adj_lists
      gathered_atoms = tf.gather(atom_features, deg_adj_lists[deg - 1])
      gathered_atoms = tf.concat(axis=1, values=[self_atoms, gathered_atoms])

      maxed_atoms = tf.reduce_max(gathered_atoms, 1)
      deg_maxed[deg - self.min_degree] = maxed_atoms

    if self.min_degree == 0:
      begin = tf.stack([deg_slice[0, 0], 0])
      size = tf.stack([deg_slice[0, 1], -1])
      self_atoms = tf.slice(atom_features, begin, size)
      deg_maxed[0] = self_atoms

    return tf.concat(axis=0, values=deg_maxed)


class GraphGather(tf.keras.layers.Layer):

  def __init__(self, batch_size, activation_fn=None, **kwargs):
    super(GraphGather, self).__init__(**kwargs)
    self.batch_size = batch_size
    self.activation_fn = activation_fn

  def call(self, inputs):
    # x = [atom_features, deg_slice, membership, deg_adj_list placeholders...]
    atom_features = inputs[0]

    # Extract graph topology
    membership = inputs[2]

    assert self.batch_size > 1, "graph_gather requires batches larger than 1"

    sparse_reps = tf.math.unsorted_segment_sum(atom_features, membership,
                                          self.batch_size)
    max_reps = tf.math.unsorted_segment_max(atom_features, membership,
                                       self.batch_size)
    mol_features = tf.concat(axis=1, values=[sparse_reps, max_reps])

    if self.activation_fn is not None:
      mol_features = self.activation_fn(mol_features)
    return mol_features


# class GraphConv(tf.keras.layers.Layer):
#     """Graph Convolutional Layers
#
#     This layer implements the graph convolution introduced in
#
#     Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints." Advances in neural information processing systems. 2015. https://arxiv.org/abs/1509.09292
#
#     The graph convolution combines per-node feature vectures in a
#     nonlinear fashion with the feature vectors for neighboring nodes.
#     This "blends" information in local neighborhoods of a graph.
#     """
#
#     def __init__(self,
#                  out_channel,
#                  min_deg=0,
#                  max_deg=10,
#                  activation_fn=None,
#                  **kwargs):
#         """Initialize a graph convolutional layer.
#
#         Parameters
#         ----------
#         out_channel: int
#           The number of output channels per graph node.
#         min_deg: int, optional (default 0)
#           The minimum allowed degree for each graph node.
#         max_deg: int, optional (default 10)
#           The maximum allowed degree for each graph node. Note that this
#           is set to 10 to handle complex molecules (some organometallic
#           compounds have strange structures). If you're using this for
#           non-molecular applications, you may need to set this much higher
#           depending on your dataset.
#         activation_fn: function
#           A nonlinear activation function to apply. If you're not sure,
#           `tf.nn.relu` is probably a good default for your application.
#         """
#         super(GraphConv, self).__init__(**kwargs)
#         self.out_channel = out_channel
#         self.min_degree = min_deg
#         self.max_degree = max_deg
#         self.activation_fn = activation_fn
#
#     def build(self, input_shape):
#         # Generate the nb_affine weights and biases
#         num_deg = 2 * self.max_degree + (1 - self.min_degree)
#         self.W_list = [
#             self.add_weight(
#                 name='kernel',
#                 shape=(int(input_shape[0][-1]), self.out_channel),
#                 initializer='glorot_uniform',
#                 trainable=True) for k in range(num_deg)
#         ]
#         self.b_list = [
#             self.add_weight(
#                 name='bias',
#                 shape=(self.out_channel,),
#                 initializer='zeros',
#                 trainable=True) for k in range(num_deg)
#         ]
#         self.built = True
#
#     def get_config(self):
#         config = super(GraphConv, self).get_config()
#         config['out_channel'] = self.out_channel
#         config['min_deg'] = self.min_degree
#         config['max_deg'] = self.max_degree
#         config['activation_fn'] = self.activation_fn
#         return config
#
#     def call(self, inputs):
#
#         # Extract atom_features
#         atom_features = inputs[0]
#
#         # Extract graph topology
#         deg_slice = inputs[1]
#         deg_adj_lists = inputs[3:]
#
#         W = iter(self.W_list)
#         b = iter(self.b_list)
#
#         # Sum all neighbors using adjacency matrix
#         deg_summed = self.sum_neigh(atom_features, deg_adj_lists)
#
#         # Get collection of modified atom features
#         new_rel_atoms_collection = (
#             self.max_degree + 1 - self.min_degree) * [None]
#
#         split_features = tf.split(atom_features, deg_slice[:, 1])
#         for deg in range(1, self.max_degree + 1):
#             # Obtain relevant atoms for this degree
#             rel_atoms = deg_summed[deg - 1]
#
#             # Get self atoms
#             self_atoms = split_features[deg - self.min_degree]
#
#             # Apply hidden affine to relevant atoms and append
#             rel_out = tf.matmul(rel_atoms, next(W)) + next(b)
#             self_out = tf.matmul(self_atoms, next(W)) + next(b)
#             out = rel_out + self_out
#
#             new_rel_atoms_collection[deg - self.min_degree] = out
#
#         # Determine the min_deg=0 case
#         if self.min_degree == 0:
#             self_atoms = split_features[0]
#
#             # Only use the self layer
#             out = tf.linalg.matmul(self_atoms, next(W)) + next(b)
#
#             new_rel_atoms_collection[0] = out
#
#         # Combine all atoms back into the list
#         atom_features = tf.concat(axis=0, values=new_rel_atoms_collection)
#
#         if self.activation_fn is not None:
#             atom_features = self.activation_fn(atom_features)
#
#         return atom_features
#
#     def sum_neigh(self, atoms, deg_adj_lists):
#         """Store the summed atoms by degree"""
#         deg_summed = self.max_degree * [None]
#
#         # Tensorflow correctly processes empty lists when using concat
#         for deg in range(1, self.max_degree + 1):
#             gathered_atoms = tf.gather(atoms, deg_adj_lists[deg - 1])
#             # Sum along neighbors as well as self, and store
#             summed_atoms = tf.math.reduce_sum(gathered_atoms, 1)
#             deg_summed[deg - 1] = summed_atoms
#
#         return deg_summed
#
#
# class GraphPool(tf.keras.layers.Layer):
#     """A GraphPool gathers data from local neighborhoods of a graph.
#
#     This layer does a max-pooling over the feature vectors of atoms in a
#     neighborhood. You can think of this layer as analogous to a max-pooling layer
#     for 2D convolutions but which operates on graphs instead.
#     """
#
#     def __init__(self, min_degree=0, max_degree=10, **kwargs):
#         """Initialize this layer
#
#         Parameters
#         ----------
#         min_deg: int, optional (default 0)
#           The minimum allowed degree for each graph node.
#         max_deg: int, optional (default 10)
#           The maximum allowed degree for each graph node. Note that this
#           is set to 10 to handle complex molecules (some organometallic
#           compounds have strange structures). If you're using this for
#           non-molecular applications, you may need to set this much higher
#           depending on your dataset.
#         """
#         super(GraphPool, self).__init__(**kwargs)
#         self.min_degree = min_degree
#         self.max_degree = max_degree
#
#     def get_config(self):
#         config = super(GraphPool, self).get_config()
#         config['min_degree'] = self.min_degree
#         config['max_degree'] = self.max_degree
#         return config
#
#     def call(self, inputs):
#         atom_features = inputs[0]
#         deg_slice = inputs[1]
#         deg_adj_lists = inputs[3:]
#
#         # Perform the mol gather
#         # atom_features = graph_pool(atom_features, deg_adj_lists, deg_slice,
#         #                            self.max_degree, self.min_degree)
#
#         deg_maxed = (self.max_degree + 1 - self.min_degree) * [None]
#
#         # Tensorflow correctly processes empty lists when using concat
#
#         split_features = tf.split(atom_features, deg_slice[:, 1])
#         for deg in range(1, self.max_degree + 1):
#             # Get self atoms
#             self_atoms = split_features[deg - self.min_degree]
#
#             if deg_adj_lists[deg - 1].shape[0] == 0:
#                 # There are no neighbors of this degree, so just create an empty tensor directly.
#                 maxed_atoms = tf.zeros((0, self_atoms.shape[-1]))
#             else:
#                 # Expand dims
#                 self_atoms = tf.expand_dims(self_atoms, 1)
#
#                 # always deg-1 for deg_adj_lists
#                 gathered_atoms = tf.gather(
#                     atom_features, deg_adj_lists[deg - 1])
#                 gathered_atoms = tf.concat(
#                     axis=1, values=[self_atoms, gathered_atoms])
#
#                 maxed_atoms = tf.rmath.educe_max(gathered_atoms, 1)
#             deg_maxed[deg - self.min_degree] = maxed_atoms
#
#         if self.min_degree == 0:
#             self_atoms = split_features[0]
#             deg_maxed[0] = self_atoms
#
#         return tf.concat(axis=0, values=deg_maxed)
#
#
# class GraphGather(tf.keras.layers.Layer):
#     """A GraphGather layer pools node-level feature vectors to create a graph feature vector.
#
#     Many graph convolutional networks manipulate feature vectors per
#     graph-node. For a molecule for example, each node might represent an
#     atom, and the network would manipulate atomic feature vectors that
#     summarize the local chemistry of the atom. However, at the end of
#     the application, we will likely want to work with a molecule level
#     feature representation. The `GraphGather` layer creates a graph level
#     feature vector by combining all the node-level feature vectors.
#
#     One subtlety about this layer is that it depends on the
#     `batch_size`. This is done for internal implementation reasons. The
#     `GraphConv`, and `GraphPool` layers pool all nodes from all graphs
#     in a batch that's being processed. The `GraphGather` reassembles
#     these jumbled node feature vectors into per-graph feature vectors.
#     """
#
#     def __init__(self, batch_size, activation_fn=None, **kwargs):
#         """Initialize this layer.
#
#         Parameters
#         ---------
#         batch_size: int
#           The batch size for this layer. Note that the layer's behavior
#           changes depending on the batch size.
#         activation_fn: function
#           A nonlinear activation function to apply. If you're not sure,
#           `tf.nn.relu` is probably a good default for your application.
#         """
#
#         super(GraphGather, self).__init__(**kwargs)
#         self.batch_size = batch_size
#         self.activation_fn = activation_fn
#
#     def get_config(self):
#         config = super(GraphGather, self).get_config()
#         config['batch_size'] = self.batch_size
#         config['activation_fn'] = self.activation_fn
#         return config
#
#     def call(self, inputs):
#         """Invoking this layer.
#
#         Parameters
#         ----------
#         inputs: list
#           This list should consist of `inputs = [atom_features, deg_slice,
#           membership, deg_adj_list placeholders...]`. These are all
#           tensors that are created/process by `GraphConv` and `GraphPool`
#         """
#         atom_features = inputs[0]
#
#         # Extract graph topology
#         membership = inputs[2]
#
#         assert self.batch_size > 1, "graph_gather requires batches larger than 1"
#
#         sparse_reps = tf.math.unsorted_segment_sum(atom_features, membership,
#                                                    self.batch_size)
#         max_reps = tf.math.unsorted_segment_max(atom_features, membership,
#                                                 self.batch_size)
#         mol_features = tf.concat(axis=1, values=[sparse_reps, max_reps])
#
#         if self.activation_fn is not None:
#             mol_features = self.activation_fn(mol_features)
#         return mol_features
#
#
# class _GraphConvKerasModel(tf.keras.Model):
#
#     def __init__(self,
#                  n_tasks,
#                  graph_conv_layers,
#                  dense_layer_size=128,
#                  dropout=0.0,
#                  number_atom_features=75,
#                  n_classes=2,
#                  batch_normalize=True,
#                  uncertainty=False,
#                  batch_size=100):
#         """An internal keras model class.
#
#         The graph convolutions use a nonstandard control flow so the
#         standard Keras functional API can't support them. We instead
#         use the imperative "subclassing" API to implement the graph
#         convolutions.
#
#         All arguments have the same meaning as in GraphConvModel.
#         """
#         super(_GraphConvKerasModel, self).__init__()
#         self.uncertainty = uncertainty
#
#         if not isinstance(dropout, collections.Sequence):
#             dropout = [dropout] * (len(graph_conv_layers) + 1)
#         if len(dropout) != len(graph_conv_layers) + 1:
#             raise ValueError('Wrong number of dropout probabilities provided')
#
#         self.graph_convs = [
#             layers.GraphConv(layer_size, activation_fn=tf.nn.relu)
#             for layer_size in graph_conv_layers
#         ]
#         self.batch_norms = [
#             BatchNormalization(fused=False) if batch_normalize else None
#             for _ in range(len(graph_conv_layers) + 1)
#         ]
#         self.dropouts = [
#             Dropout(rate=rate) if rate > 0.0 else None for rate in dropout
#         ]
#         self.graph_pools = [layers.GraphPool() for _ in graph_conv_layers]
#         self.dense = Dense(dense_layer_size, activation=tf.nn.relu)
#         self.graph_gather = layers.GraphGather(
#             batch_size=batch_size, activation_fn=tf.nn.tanh)
#         self.trim = TrimGraphOutput()
#         self.regression_dense = Dense(n_tasks)
#         if self.uncertainty:
#             self.uncertainty_dense = Dense(n_tasks)
#             self.uncertainty_trim = TrimGraphOutput()
#             self.uncertainty_activation = Activation(tf.exp)
#
#     def call(self, inputs, training=False):
#         atom_features = inputs[0]
#         degree_slice = tf.cast(inputs[1], dtype=tf.int32)
#         membership = tf.cast(inputs[2], dtype=tf.int32)
#         n_samples = tf.cast(inputs[3], dtype=tf.int32)
#         deg_adjs = [tf.cast(deg_adj, dtype=tf.int32) for deg_adj in inputs[4:]]
#
#         in_layer = atom_features
#         for i in range(len(self.graph_convs)):
#             gc_in = [in_layer, degree_slice, membership] + deg_adjs
#             gc1 = self.graph_convs[i](gc_in)
#             if self.batch_norms[i] is not None:
#                 gc1 = self.batch_norms[i](gc1, training=training)
#             if training and self.dropouts[i] is not None:
#                 gc1 = self.dropouts[i](gc1, training=training)
#             gp_in = [gc1, degree_slice, membership] + deg_adjs
#             in_layer = self.graph_pools[i](gp_in)
#         dense = self.dense(in_layer)
#         if self.batch_norms[-1] is not None:
#             dense = self.batch_norms[-1](dense, training=training)
#         if training and self.dropouts[-1] is not None:
#             dense = self.dropouts[1](dense, training=training)
#         neural_fingerprint = self.graph_gather([dense, degree_slice, membership] +
#                                                deg_adjs)
#         output = self.regression_dense(neural_fingerprint)
#         output = self.trim([output, n_samples])
#         if self.uncertainty:
#             log_var = self.uncertainty_dense(neural_fingerprint)
#             log_var = self.uncertainty_trim([log_var, n_samples])
#             var = self.uncertainty_activation(log_var)
#             outputs = [output, var, output, log_var, neural_fingerprint]
#         else:
#             outputs = [output, neural_fingerprint]
#
#         return outputs
#
#
# class GraphConvModel(KerasModel):
#     """Graph Convolutional Models.
#
#     This class implements the graph convolutional model from the
#     following paper:
#
#
#     Duvenaud, David K., et al. "Convolutional networks on graphs for learning molecular fingerprints." Advances in neural information processing systems. 2015.
#
#     """
#
#     def __init__(self,
#                  n_tasks,
#                  graph_conv_layers=[64, 64],
#                  dense_layer_size=128,
#                  dropout=0.0,
#                  number_atom_features=75,
#                  n_classes=2,
#                  batch_size=100,
#                  batch_normalize=True,
#                  uncertainty=False,
#                  **kwargs):
#         """The wrapper class for graph convolutions.
#
#         Note that since the underlying _GraphConvKerasModel class is
#         specified using imperative subclassing style, this model
#         cannout make predictions for arbitrary outputs.
#
#         Parameters
#         ----------
#         n_tasks: int
#           Number of tasks
#         graph_conv_layers: list of int
#           Width of channels for the Graph Convolution Layers
#         dense_layer_size: int
#           Width of channels for Atom Level Dense Layer before GraphPool
#         dropout: list or float
#           the dropout probablity to use for each layer.  The length of this list should equal
#           len(graph_conv_layers)+1 (one value for each convolution layer, and one for the
#           dense layer).  Alternatively this may be a single value instead of a list, in which
#           case the same value is used for every layer.
#         mode: str
#           Either "classification" or "regression"
#         number_atom_features: int
#             75 is the default number of atom features created, but
#             this can vary if various options are passed to the
#             function atom_features in graph_features
#         n_classes: int
#           the number of classes to predict (only used in classification mode)
#         batch_normalize: True
#           if True, apply batch normalization to model
#         uncertainty: bool
#           if True, include extra outputs and loss terms to enable the uncertainty
#           in outputs to be predicted
#         """
#         self.n_tasks = n_tasks
#         self.n_classes = n_classes
#         self.batch_size = batch_size
#         self.uncertainty = uncertainty
#         model = _GraphConvKerasModel(
#             n_tasks,
#             graph_conv_layers=graph_conv_layers,
#             dense_layer_size=dense_layer_size,
#             dropout=dropout,
#             number_atom_features=number_atom_features,
#             n_classes=n_classes,
#             batch_normalize=batch_normalize,
#             uncertainty=uncertainty,
#             batch_size=batch_size)
#         if self.uncertainty:
#             output_types = ['prediction', 'variance',
#                             'loss', 'loss', 'embedding']
#
#             def loss(outputs, labels, weights):
#                 diff = labels[0] - outputs[0]
#                 return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
#         else:
#             output_types = ['prediction', 'embedding']
#             loss = L2Loss()
#         super(GraphConvModel, self).__init__(
#             model, loss, output_types=output_types, batch_size=batch_size, **kwargs)
#
#     def default_generator(self,
#                           dataset,
#                           epochs=1,
#                           mode='fit',
#                           deterministic=True,
#                           pad_batches=True):
#         for epoch in range(epochs):
#             for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
#                 batch_size=self.batch_size,
#                 deterministic=deterministic,
#                     pad_batches=pad_batches):
#                 multiConvMol = ConvMol.agglomerate_mols(X_b)
#                 n_samples = np.array(X_b.shape[0])
#                 inputs = [
#                     multiConvMol.get_atom_features(), multiConvMol.deg_slice,
#                     np.array(multiConvMol.membership), n_samples
#                 ]
#                 for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
#                     inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
#                 yield (inputs, [y_b], [w_b])
