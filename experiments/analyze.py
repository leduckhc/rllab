import argparse
import os
import joblib
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import lasagne.layers as L
import theano
import theano.tensor as T
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product

from rllab.envs.gym_env import GymEnv
from rllab.envs.preprocess_env import PreprocessEnv
from rllab.envs.sliding_mem_env import SlidingMemEnv
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.console import query_yes_no, mkdir_p
from rllab.sampler.utils import rollout


def save_plt_to_pdf(plt_c, log_file):
    """
    Parameters
    ----------
    plt_c : matplotlib.fig
    log_file : without extension
    """
    global analysis_log_dir

    log_file = os.path.join(analysis_log_dir, log_file + '.pdf')
    with PdfPages(log_file) as pdf:
        for i in plt_c.get_fignums():
            fig = plt_c.figure(i)
            pdf.savefig(fig)
    plt_c.close('all')


def visualize_conv_weight(c_layer, figsize=(6, 6), file_name=None):
    """
    Visualize weights of conv layer. Inspired by nolearn.visualize.
    Parameters
    ----------
    c_layer : lasagne.layers.conv.Conv2DLayer
    figsize :
    file_name : str
    """
    W = c_layer.W.get_value()
    # shape = (num_filters, num_input_channels, filter_rows, filter_columns)
    shape = W.shape
    rows = np.ceil(np.sqrt(shape[0])).astype(int)  # num_filters
    cols = rows

    for channel in range(shape[1]):
        figs, axes = plt.subplots(rows, cols, squeeze=False)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        for i, (r, c) in enumerate(product(range(rows), range(cols))):
            if i >= shape[0]:
                break
            img = W[i, channel]
            axes[r, c].imshow(img, vmin=img.min(), vmax=img.max(),
                              interpolation='none', cmap='gray')

    save_plt_to_pdf(plt, file_name)


def visualize_conv_activation(c_layer, x, figsize=(6, 6), file_name=None):
    """

    Parameters
    ----------
    c_layer : lasagne.layers.conv.Conv2DLayer
    x : numpy.ndarray
        Input the `conv_layer` has shape BC01 (batch, channels, h, w)
    figsize :
    """
    # compile theano function
    xs = T.tensor4('xs').astype(theano.config.floatX)
    get_activity = theano.function([xs], L.get_output(c_layer, xs, deterministic=True))

    activity = get_activity(x)
    # shape = (num_filters, num_input_channels, filter_rows, filter_columns)
    shape = activity.shape
    nrows = np.ceil(np.sqrt(shape[1])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows + 1, ncols, squeeze=False)
    num_chanels = c_layer.W.get_value().shape[1]

    for chan in range(num_chanels):
        axes[0][chan].imshow(x[0][chan], cmap='gray', interpolation='none')
        axes[0][chan].set_title('original %d-channel' % chan)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[1]:
            break
        ndim = activity[0][i].ndim
        if ndim != 2:
            raise ValueError("Wrong number of dimensions, image data should "
                             "have 2, instead got {}".format(ndim))
        axes[r + 1, c].imshow(activity[0][i], cmap='gray',
                              interpolation='none')

    save_plt_to_pdf(plt, file_name)


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str,
                    help='Path to the snapshot file. Usually it is ".pkl" file')
parser.add_argument('--env', type=str, help='Name of the environemtn')
parser.add_argument('--visualize_conv', type=bool, default=True, help='Visualize convolution layer')
args = parser.parse_args()

analysis_log_dir = os.path.join('analysis', args.env)
mkdir_p(analysis_log_dir)


# If the snapshot file use tensorflow, do:
# import tensorflow as tf
# with tf.Session():
#     [rest of the code]
with tf.Session() as sess:
    data = joblib.load(args.file)
    policy = data['policy']
    if args.env:
        agent_history_length = 4
        resized_shape = (84, 84)
        env = GymEnv(env_name=args.env, record_video=True, log_dir=analysis_log_dir, record_log=True, force_reset=True)
        env = PreprocessEnv(env, new_shape=resized_shape)
        env = SlidingMemEnv(env, n_steps=agent_history_length)
    else:
        env = data['env']

    if args.visualize_conv:
        path = rollout(env, policy, animated=False)
        observations, actions, rewards = ext.extract(
            path,
            "observations", "actions", "rewards"
        )

        logger.log("Generating plots of networks")
        conv_layer = L.get_all_layers(policy._output_layer)[2]
        visualize_conv_weight(conv_layer, file_name='vis_weights')
        path_length = len(observations)
        some_indexes = np.linspace(0, path_length, 10, endpoint=False, dtype='int32')
        for obs_ind in some_indexes:
            obs = observations[obs_ind].reshape(1,4,84,84)
            visualize_conv_activation(conv_layer, obs, file_name='vis_activations_%d' % obs_ind)
        logger.log("Finished plotting")

    while True:
        path = rollout(env, policy, animated=True, max_path_length=5000)
        if not query_yes_no('Continue simulation?'):
            break
