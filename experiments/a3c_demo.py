from experiments.a3c import A3C
from rllab.envs.gym_env import GymEnv
from rllab.envs.preprocess_env import PreprocessEnv
from rllab.envs.sliding_mem_env import SlidingMemEnv
from rllab.exploration_strategies.epsilon_strategy import EpsilonGreedyStrategy
from rllab.misc.instrument import stub
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from rllab.v_functions.continuous_mlp_v_function import ContinuousMLPVFunction

stub(globals())

env = GymEnv("SpaceInvaders-v0")
env = PreprocessEnv(env, new_shape=(84,84))
env = SlidingMemEnv(env, n_steps=4)

policy = CategoricalConvPolicy(
    env_spec=env.spec)
vfunc = ContinuousMLPVFunction(
    env_spec=env.spec)
es = EpsilonGreedyStrategy(
    env_spec=env.spec)

algo = A3C(
    env = env,
    policy=policy,
    vfunc=vfunc,
    es=es,
)

