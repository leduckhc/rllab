from experiments.a3c import A3C
from rllab.envs.gym_env import GymEnv
from rllab.envs.preprocess_env import PreprocessEnv
from rllab.envs.sliding_mem_env import SlidingMemEnv
from rllab.exploration_strategies.epsilon_strategy import EpsilonGreedyStrategy
from rllab.misc.instrument import stub
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from rllab.v_functions.continuous_conv_v_function import ContinuousConvVFunction
import lasagne.nonlinearities as NL


env = GymEnv("Pong-v0")
env = PreprocessEnv(env, new_shape=(84, 84))
env = SlidingMemEnv(env, n_steps=4)

policy = CategoricalConvPolicy(
    env_spec=env.spec,
    conv_filters=(32, 64, 64),
    conv_filter_sizes=(8, 4, 3),
    conv_strides=(4, 4, 2),
    conv_pads=('valid', 'valid', 'valid'),
    hidden_sizes=[512],
    hidden_nonlinearity=NL.rectify,
    output_nonlinearity=NL.linear,
    name="categorical_conv_policy"
)

vfunc = ContinuousConvVFunction(
    env_spec=env.spec,
    conv_filters=(32, 64, 64),
    conv_filter_sizes=(8, 4, 3),
    conv_strides=(4, 4, 2),
    conv_pads=('valid', 'valid', 'valid'),
    hidden_sizes=[512],
    hidden_nonlinearity=NL.rectify,
    output_nonlinearity=NL.linear,
    # name="continuous_conv_v_func"
)

es = EpsilonGreedyStrategy(
    env_spec=env.spec)

algo = A3C(
    env=env,
    policy=policy,
    vfunc=vfunc,
    es=es,
    n_epochs=3,
    epoch_length=1000
)

algo.train()


