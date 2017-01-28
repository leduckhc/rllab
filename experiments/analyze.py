import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout


# logger.log("Generating plots of networks")
# self.policy.visualize_conv_weight(epoch)
# best_return_ind = np.argmax(returns)
# some_obs_length = len(paths[best_return_ind]['observations'])
# some_obs_ind = np.linspace(0, some_obs_length, 10, endpoint=False, dtype=np.intp)
# some_obs = paths[best_return_ind]['observations'][some_obs_ind]
# self.policy.visualize_conv_activation(some_obs, epoch)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='Path to the snapshot file. Usually it is ".pkl" file')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup)
            if not query_yes_no('Continue simulation?'):
                break

