from experiments.dummy_algo import DummyAlgo
from rllab.algos.vpg import VPG
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.misc.instrument import run_experiment_lite, stub
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.uniform_control_policy import UniformControlPolicy

from rllab.sampler import parallel_sampler

stub(globals())

env = GymEnv("SpaceInvaders-v0", force_reset=True)
# env = AtariEnvWrapper(env, 4, 84, 84)
# env = CartpoleEnv()
policy = CategoricalMLPPolicy(env.spec)
baseline = ZeroBaseline(env.spec)
# parallel_sampler.initialize(n_parallel=1)
algo = VPG(env, policy, baseline)
algo = DummyAlgo(env, policy)
# algo.train()


run_experiment_lite(
    algo.train(),
    exp_prefix="dummy-tester",
    # Number of parallel workers for sampling
    n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # mode="ec2",
    # use_gpu=True,
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
    # sync_s3_pkl=True,
)

