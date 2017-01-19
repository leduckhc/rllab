from experiments.dqn.dqn import DQN
from rllab.envs.gym_env import GymEnv
from rllab.misc.instrument import run_experiment_lite, stub

stub(globals())

#
# env = GymEnv("SpaceInvaders-v0")
# algo = DQN(
#     env,
#     n_epochs=1000,
#     batch_size=32,
#     discount=0.99,
#     replay_memory_size=1000000, #1000000
#     min_replay_memory_size=50000,
#     epoch_length=50000,
#     max_path_length=10000,
#     target_network_update_frequency=10000,
#     agent_history_length=4
# )

env = GymEnv("SpaceInvaders-v0")
algo = DQN(
    env,
    n_epochs=10,
    batch_size=32,
    discount=0.99,
    replay_memory_size=1000, #1000000
    min_replay_memory_size=5,
    epoch_length=50,
    max_path_length=10,
    target_network_update_frequency=10,
    agent_history_length=4
)



run_experiment_lite(
    algo.train(),
    exp_prefix="space-invaders",
    exp_name="dqn",
    # Number of parallel workers for sampling
    n_parallel=8,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="all",
    mode="local",
    use_gpu=False,
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
    sync_s3_pkl=True,
    periodic_sync=True,
    periodic_sync_interval=120,
    sync_log_on_termination=True,
)
