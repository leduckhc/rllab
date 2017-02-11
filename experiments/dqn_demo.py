import argparse

import lasagne.nonlinearities as NL
import sys

from rllab.algos.dqn import DQN
from rllab.envs.gym_env import GymEnv
from rllab.envs.preprocess_env import PreprocessEnv
from rllab.envs.sliding_mem_env import SlidingMemEnv
from rllab.exploration_strategies.epsilon_strategy import EpsilonGreedyStrategy
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.deterministic_conv_policy import DeterministicConvPolicy

env_name = 'SpaceInvaders-v0'
n_epochs = 200
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default=env_name,
                    help='Name of the environment. Deefault value is SpaceInvaders-v0')
parser.add_argument('--n_epochs', type=int, default=n_epochs,
                    help='Number of epochs. Default value is 200')
args = parser.parse_args(sys.argv[1:])

if args.env:
    env_name = args.env
if args.n_epochs:
    n_epochs = args.n_epochs


def run_task(*_):
    agent_history_length = 4
    resized_shape = (84, 84)

    env = GymEnv(env_name=env_name, record_video=False, record_log=False, force_reset=True)
    env = PreprocessEnv(env, new_shape=resized_shape)
    env = SlidingMemEnv(env, n_steps=agent_history_length)

    policy = DeterministicConvPolicy(
        env_spec=env.spec,
        conv_filters=(32, 64, 64),
        conv_filter_sizes=(8, 4, 3),
        conv_strides=(4, 2, 1),
        conv_pads=('valid', 'valid', 'valid'),
        hidden_sizes=[512],
        hidden_nonlinearity=NL.rectify,
        output_nonlinearity=NL.linear
    )

    es = EpsilonGreedyStrategy(
        env_spec=env.spec,
        eps_start=1.0,
        eps_final=0.1,
        eps_itr_start=0,  # 200,
        eps_itr_final=1000000,
    )

    algo_lite = DQN(
        env,
        policy,
        es,
        n_epochs=n_epochs,
        epoch_length=1000,  # 1000,
        batch_size=2,
        discount=0.99,
        replay_memory_size=20000,  # 20000 #10^5=11gb
        min_replay_memory_size=5000,  # 2000, #50000
        target_network_update_frequency=500,  # 500, #10000,
        agent_history_length=agent_history_length,
        resized_shape=resized_shape,
        eval_max_samples=5000,  # 10000,#50000,
        eval_max_path_length=500,  # 1000,s
        update_method='rmsprop',
        update_method_kwargs=dict(
            learning_rate=0.00025, rho=0.95, epsilon=1e-2),
        # plot=True,
    )

    algo = DQN(
        env,
        policy,
        es,
        n_epochs=n_epochs,
        epoch_length=50000, #1000,
        batch_size=32,
        discount=0.99,
        replay_memory_size=1000000, #20000 #10^5=11gb
        min_replay_memory_size=50000, #2000, #50000
        target_network_update_frequency=10000, #500, #10000,
        agent_history_length=agent_history_length,
        resized_shape=resized_shape,
        eval_max_samples=100000,#10000,#50000,
        eval_max_path_length=2000,#1000,s
        # plot=True,
    )

    # TODO: dont forget to train your algorithm
    algo_lite.train()


# run experiment
run_experiment_lite(
    run_task,
    exp_prefix=env_name,
    exp_name="dqn",
    # Number of parallel workers for sampling.
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="all",
    mode="local",
    use_gpu=True,  # TODO True
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=717836484,
    # plot=True,
)
