import cv2

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides
from rllab.spaces import Box


class PreprocessEnv(ProxyEnv, Serializable):
    """
    Transform an observation to grayscale and resize it. Yet works for Atari games
    """

    def __init__(self, env, new_shape=(84, 84)):
        assert isinstance(env.observation_space, Box)

        super(PreprocessEnv, self).__init__(env)
        Serializable.quick_init(self, locals())
        self.new_shape = new_shape
        self.range = env.observation_space.high - env.observation_space.low

    def preprocess(self, obs):
        # scale down game image
        # set the grayscale to have values in the 0.0 to 1.0 range
        return cv2.cvtColor(cv2.resize(src=obs, dsize=self.new_shape), cv2.COLOR_BGR2GRAY) / 255.0

    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        return self.preprocess(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        return Step(self.preprocess(next_obs), reward, done, **info)
