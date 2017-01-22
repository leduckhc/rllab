from rllab.algos.dqn import DQN
from rllab.envs.gym_env import GymEnv
from rllab.misc.instrument import run_experiment_lite, stub

stub(globals())

env = GymEnv("SpaceInvaders-v0")
algo = DQN(
    env,
    n_epochs=1000,
    batch_size=32,
    discount=0.99,
    replay_memory_size=100000, #1000000
    min_replay_memory_size=20000,
    epoch_length=20000,
    max_path_length=5000,
    target_network_update_frequency=10000,
    agent_history_length=4
)

run_experiment_lite(
    algo.train(),
    exp_prefix="space-invaders",
    exp_name="dqn",
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="all",
    mode="ec2",
    use_gpu=True,
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=3,
    # plot=True,
    sync_s3_pkl=True,
    periodic_sync=True,
    # syncing frequency in seconds
    periodic_sync_interval=15,
    sync_log_on_termination=True,
)
