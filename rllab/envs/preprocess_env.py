import cv2
import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides
from rllab.spaces import Box



class PreprocessEnv(ProxyEnv, Serializable):
    """
    Transform an observation to grayscale and resize it. Yet works for Atari games
    """

    def __init__(self, env, new_shape=(84, 84), insert_extra_one_to_shape=True):
        assert isinstance(env.observation_space, Box)
        # element on last index is 3 is BGR or RGB
        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3

        super(PreprocessEnv, self).__init__(env)
        Serializable.quick_init(self, locals())
        self.new_shape = new_shape
        self.range = np.max(env.observation_space.high - env.observation_space.low)
        self.insert_extra_one_to_shape = insert_extra_one_to_shape

    def preprocess(self, obs):
        # scale down game image
        # set the grayscale to have values in the 0.0 to 1.0 range
        # returned shape is of new_shape
        return cv2.cvtColor(cv2.resize(obs, self.new_shape), cv2.COLOR_BGR2GRAY) / self.range

    @property
    def observation_space(self):
        if self.insert_extra_one_to_shape:
            # we need (1,) + self.new_shape because new_shape==(84,84)
            return Box(0, 1, shape=(1,) + self.new_shape)
        else:
            return Box(0, 1, shape=self.new_shape)

    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        return self.preprocess(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        return Step(self.preprocess(next_obs), reward, done, **info)
