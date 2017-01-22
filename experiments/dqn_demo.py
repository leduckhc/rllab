import argparse

import lasagne.nonlinearities as NL
import sys

from rllab.algos.dqn import DQN
from rllab.envs.gym_env import GymEnv
from rllab.envs.preprocess_env import PreprocessEnv
from rllab.envs.sliding_mem_env import SlidingMemEnv
from rllab.exploration_strategies.epsilon_strategy import EpsilonGreedyStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.policies.deterministic_conv_policy import DeterministicConvPolicy


if __name__ == '__main__':
    default_env_name = 'SpaceInvaders-v0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=default_env_name, help='Name of the environment. Deefault value is SpaceInvaders-v0')
    args = parser.parse_args(sys.argv[1:])

    env_name = default_env_name
    if args.env:
        env_name = args.env

    def run_task(*_):

        agent_history_length = 4
        resized_shape = (84,84)

        env = GymEnv(env_name=env_name, record_video=False, record_log=False, force_reset=True)
        env = PreprocessEnv(env, new_shape=resized_shape)
        env = SlidingMemEnv(env, n_steps=agent_history_length)

        policy = DeterministicConvPolicy(
            env_spec=env.spec,
            conv_filters=(16,32,32),
            conv_filter_sizes=(8,4,4),
            conv_strides=(4,4,2),
            conv_pads=('valid','valid','valid'),
            hidden_sizes=[256],
            hidden_nonlinearity=NL.rectify,
            output_nonlinearity=NL.linear
        )

        es = EpsilonGreedyStrategy(
            env_spec=env.spec,
            eps_start=1.0,
            eps_final=0.1,
            eps_itr_start=10000, #200,
            eps_itr_final=1000000,
        )

        algo = DQN(
            env,
            policy,
            es,
            n_epochs=500,
            epoch_length=20000, #1000,
            batch_size=32,
            discount=0.99,
            replay_memory_size=1000000, #20000 #10^5=11gb
            min_replay_memory_size=50000, #2000, #50000
            target_network_update_frequency=10000, #500, #10000,
            agent_history_length=agent_history_length,
            resized_shape=resized_shape,
            eval_max_samples=50000,#10000,#50000,
            eval_max_path_length=1000,#1000,s
            # plot=True,
        )
        algo.train()

    run_experiment_lite(
        run_task(),
        exp_prefix=env_name,
        exp_name="dqn",
        # Number of parallel workers for sampling.
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        mode="local",
        use_gpu=True, # TODO True
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )