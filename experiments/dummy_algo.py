from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from rllab.envs import gym_env
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler
from rllab.policies.uniform_control_policy import UniformControlPolicy
from rllab.misc import logger

class DummyAlgo(RLAlgorithm, Serializable):
    def __init__(self, env, policy, scope=None):
        print("Init..")
        Serializable.quick_init(self, locals())

        self.env = env
        self.policy = policy
        self.scope = scope

    @overrides
    def train(self):
        parallel_sampler.populate_task(self.env, self.policy, self.scope)
        obs = self.env.reset()
        for i in range(10):
            logger.log("Epoch % d" % i)
            for _ in range(100):
                action, _ = self.policy.get_action(obs)
                next_obs, rew, done, info = self.env.step(action)
                obs = next_obs if not done else self.env.reset()
            logger.log("Evaluating...")
            paths = parallel_sampler.sample_paths(
                policy_params=self.policy.get_param_values(),
                max_samples=20,
                max_path_length=100,
            )
        parallel_sampler.terminate_task()

    def get_itr_snapshot(self):
        pass
