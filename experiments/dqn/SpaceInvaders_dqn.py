import lasagne.nonlinearities as NL

from rllab.algos.dqn import DQN
from rllab.envs.gym_env import GymEnv
from rllab.envs.preprocess_env import PreprocessEnv
from rllab.envs.sliding_mem_env import SlidingMemEnv
from rllab.exploration_strategies.epsilon_strategy import EpsilonGreedyStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.policies.deterministic_conv_policy import DeterministicConvPolicy

stub(globals())

agent_history_length = 4
resized_shape = (84,84)

env = GymEnv("SpaceInvaders-v0")
env = PreprocessEnv(env, new_shape=resized_shape)
env = SlidingMemEnv(env, n_steps=agent_history_length)

policy = DeterministicConvPolicy(
    env_spec=env.spec,
    conv_filters=(16,32,32),
    conv_filter_sizes=(8,4,4),
    conv_strides=(4,4,2),
    hidden_sizes=[256],
    hidden_nonlinearity=NL.rectify,
    output_nonlinearity=NL.linear
)

es = EpsilonGreedyStrategy(
    env_spec=env.spec,
    eps_start=1.0,
    eps_final=0.1,
    eps_itr_start=10000,
    eps_itr_final=1000000,
)

algo = DQN(
    env,
    policy,
    es,
    n_epochs=200,
    epoch_length=50000,
    batch_size=32,
    discount=0.99,
    replay_memory_size=2000, #200000
    min_replay_memory_size=100, #10000
    target_network_update_frequency=10000,
    agent_history_length=4,
    eval_max_samples=50000,
    eval_max_path_length=1000,
    # plot=False,
)

run_experiment_lite(
    algo.train(),
    exp_prefix="space-invaders",
    exp_name="dqn-small",
    # Number of parallel workers for sampling.
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    mode="local",
    use_gpu=True, # TODO True
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=3,
    # plot=True,
)
