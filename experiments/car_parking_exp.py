import sys
from collections import OrderedDict

import gym

from rllab.algos.cem import CEM
from rllab.algos.cma_es import CMAES
from rllab.algos.ddpg import DDPG
from rllab.algos.erwr import ERWR
from rllab.algos.power import POWER
from rllab.algos.ppo import PPO
from rllab.algos.reps import REPS
from rllab.algos.tnpg import TNPG
from rllab.algos.trpo import TRPO
from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.car_parking_env import CarParkingEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.ext import lazydict
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

stub(globals())

for (algo_cls, algo_name) in zip([VPG, ERWR, POWER, TNPG, PPO, REPS, CEM],
                                 ['VPG', 'ERWR', 'POWER', 'TNPG', 'PPO', 'REPS', 'CEM']):
    env = normalize(CarParkingEnv())
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    common_batch_algo_args = dict(
        n_itr=200,
        batch_size=4000,
        max_path_length=4000,
        env=env, policy=policy, baseline=baseline,
    )
    
    algos = {
        VPG: common_batch_algo_args,
        ERWR: common_batch_algo_args,
        POWER: common_batch_algo_args,
        TNPG: common_batch_algo_args,
        PPO: common_batch_algo_args,
        REPS: common_batch_algo_args,
        CEM: common_batch_algo_args,
        CMAES: common_batch_algo_args,
        DDPG: dict(
            env=env,
            policy=DeterministicMLPPolicy(env.spec),
            qf=ContinuousMLPQFunction(env.spec),
            es=OUStrategy(env.spec),
            n_epochs=100,
            epoch_length=4000,
            batch_size=32,
            min_pool_size=5000,
            replay_pool_size=100000,
            eval_samples=200,
        ),
    }
    
    algo = algo_cls(**algos[algo_cls])
    algo.train()
    
    run_experiment_lite(
        algo.train(),
        exp_prefix="car-parking-exp",
        exp_name="exp-%s" % algo_name,
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        # snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # mode="local",
        #############################3
        mode="ec2",
        terminate_machine=True,
        periodic_sync=True,
        periodic_sync_interval=120,
        sync_s3_pkl=True,
        snapshot_mode='all',
        # use_gpu=True,
    )
    
    # sys.exit()
