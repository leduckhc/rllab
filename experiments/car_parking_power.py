from rllab.algos.power import POWER
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

stub(globals())

env = normalize(CartpoleEnv())
baseline = GaussianMLPBaseline(env.spec)
policy = GaussianMLPPolicy(env.spec)

algo = POWER(
    env=env,
    baseline=baseline,
    policy=policy,
    n_itr=2,
    batch_size=5000,
    max_path_length=500,
)

run_experiment_lite(
    algo.train(),
    exp_prefix="car_parking_exp",
    exp_name="exp_POWER",
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    mode="local",
    # mode="ec2",
    # variant=v,
    # plot=True,
    # terminate_machine=False,
)
