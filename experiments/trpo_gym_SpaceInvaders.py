# from rllab.algos.dqn import DQN
# from rllab.misc.instrument import stub, run_experiment_lite
# from rllab.envs.gym_env import GymEnv
# from rllab.envs.normalized_env import normalize
# import gym
#
# algo = DQN(env=env, num_episodes=20, state_size=env.observation_space.shape,
#            number_of_actions=env.action_space.n,
#            save_name=env_name)

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import *
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer




stub(globals())

env = GymEnv("SpaceInvaders-v0")

policy = CategoricalConvPolicy(
    env_spec=env.spec,
    conv_filters=(32, 32),
    conv_filter_sizes=(2, 2),
    conv_strides=(1, 1),
    conv_pads=(0, 0),
    hidden_sizes=(10, 10),
)

baseline = ZeroBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=1000,
    max_path_length=env.horizon,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
    #optimizer_args=dict(num_slices=2),
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

run_experiment_lite(
    algo.train(),
    exp_prefix="trpo-space-invaders",
    # Number of parallel workers for sampling
    n_parallel=4,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="all",
    mode="local",
    # mode="ec2",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
    # use_cloudpickle=True,
    # use_gpu=True,
)
