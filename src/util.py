# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/uncertainty/blob/master/LICENSE.

# Dependency imports
from absl import app
from absl import flags
import matplotlib
matplotlib.use('Agg')
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

def plot_weight_posteriors(names, qm_vals, qs_vals, fname):
  """Save a PNG plot with histograms of weight means and stddevs.

  Args:
    names: A Python `iterable` of `str` variable names.
      qm_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior means of weight varibles.
    qs_vals: A Python `iterable`, the same length as `names`,
      whose elements are Numpy `array`s, of any shape, containing
      posterior standard deviations of weight varibles.
    fname: Python `str` filename to save the plot to.
  """
  fig = figure.Figure(figsize=(6, 3))
  canvas = backend_agg.FigureCanvasAgg(fig)

  ax = fig.add_subplot(1, 2, 1)
  for n, qm in zip(names, qm_vals):
    sns.distplot(tf.reshape(qm, shape=[-1]), ax=ax, label=n)
  ax.set_title('weight means')
  ax.set_xlim([-1.5, 1.5])
  ax.legend()

  ax = fig.add_subplot(1, 2, 2)
  for n, qs in zip(names, qs_vals):
    sns.distplot(tf.reshape(qs, shape=[-1]), ax=ax)
  ax.set_title('weight stddevs')
  ax.set_xlim([0, 1.])

  fig.tight_layout()
  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))


def plot_heldout_prediction(input_vals, probs,
                            fname, n=10, title=''):
  """Save a PNG plot visualizing posterior uncertainty on heldout data.

  Args:
    input_vals: A `float`-like Numpy `array` of shape
      `[num_heldout] + IMAGE_SHAPE`, containing heldout input images.
    probs: A `float`-like Numpy array of shape `[num_monte_carlo,
      num_heldout, num_classes]` containing Monte Carlo samples of
      class probabilities for each heldout sample.
    fname: Python `str` filename to save the plot to.
    n: Python `int` number of datapoints to vizualize.
    title: Python `str` title for the plot.
  """
  fig = figure.Figure(figsize=(9, 3*n))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 3*i + 1)
    ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]), interpolation='None')

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.1, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title('posterior samples')

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), tf.reduce_mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('predictive probs')
  fig.suptitle(title)
  fig.tight_layout()

  canvas.print_figure(fname, format='png')
  print('saved {}'.format(fname))

# function for plotting training via deep ensembles
def plot_training_deep_ens():
    preds, sigmas = [], []
    for j in range(len(train_x)):
        mu, sigma = get_intermediate([[train_x[j]]])
        preds.append(mu.reshape(1,)[0])
        sigmas.append(sigma.reshape(1,)[0])plt.figure(1, figsize=(15, 9))

    plt.plot([i[0] for i in train_x], [i for i in train_y])
    plt.plot([i[0] for i in train_x], [i for i in preds], 'b', linewidth=3)
    upper = [i+k for i,k in zip(preds, sigmas)]
    lower = [i-k for i,k in zip(preds, sigmas)]plt.plot([i[0] for i in train_x], [i for i in upper], 'r', linewidth = 3)
    plt.plot([i[0] for i in train_x], [i for i in lower], 'r', linewidth = 3)
    plt.plot([i[0] for i in train_x], [pow_fun(i[0]) for i in train_x], 'y', linewidth = 2)

# function for plotting testing via deep ensembles
def plot_test_deep_ens():
    x_ax = np.linspace(-4, 4, num=200)
    preds, sigmas = [], []
    for j in range(len(x_ax)):
        mu, sigma = get_intermediate([[np.array([x_ax[j]])]])
        preds.append(mu.reshape(1,)[0])
        sigmas.append(sigma.reshape(1,)[0])plt.figure(1, figsize=(15, 9))

    plt.plot([i for i in x_ax], [i for i in preds], 'b', linewidth=3)
    upper = [i+k for i,k in zip(preds, sigmas)]
    lower = [i-k for i,k in zip(preds, sigmas)]plt.plot([i for i in x_ax], [i for i in upper], 'r', linewidth = 3)
    plt.plot([i for i in x_ax], [i for i in lower], 'r', linewidth = 3)
    plt.plot([i for i in x_ax], [pow_fun(i) for i in x_ax], 'y', linewidth = 2)
