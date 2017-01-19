from collections import deque

import cv2
import numpy as np
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.gym_env import GymEnv
from rllab.spaces import Box


class AtariEnvWrapper(Env, Serializable):
    def __init__(self,
                 gym_env,
                 agent_history_length=4,
                 resized_height=84,
                 resized_width=84
        ):
        assert isinstance(gym_env, GymEnv)
        assert isinstance(gym_env.observation_space, Box)

        Serializable.quick_init(self, locals())

        self._env = gym_env
        self._resized_height = resized_height
        self._resized_width = resized_width
        self._agent_history_length = agent_history_length
        self._observation_space = Box(0, 255, (agent_history_length, resized_width, resized_height))
        self._action_space = gym_env.action_space

        self._state_buffer = deque()
        self._last_frame = None

    ###############################################################
    ## We change these methods

    def preprocess_frame(self, obs):
        """

        Parameters
        ----------
        obs :
            Frame of format (height, width, channel)

        Returns
        -------
        np.array of shape (resized_width, resized_height)
            Transform image to grayscale and then rescale it. Based on the paper Mnih et al.
            And due to flickering of some envs we take maximum of the frame and last frame.
        """
        # scale down game image
        cur_frame = cv2.cvtColor(cv2.resize(obs, (self._resized_height, self._resized_width)), cv2.COLOR_BGR2GRAY)
        # set the grayscale to have values in the 0.0 to 1.0 range
        cur_frame = cur_frame / 255.0

        last_frame = self._last_frame
        self._last_frame = cur_frame
        if last_frame is None:
            return cur_frame
        else:
            return np.maximum(cur_frame, last_frame)

    def step(self, action):
        x0, r0, done, info = self._env.step(action)
        x0 = self.preprocess_frame(x0)
        s0 = np.array(self._state_buffer)  # is correct len=anget_history_length-1
        self._state_buffer.append(x0)
        self._state_buffer.popleft()
        return s0, r0, done, info

    def reset(self):
        self._state_buffer = deque()
        x0 = self._env.reset()
        x0 = self._last_frame = self.preprocess_frame(x0)
        self._state_buffer.extend((x0,) * self._agent_history_length)
        s0 = np.stack((x0,) * self._agent_history_length, axis=0)
        return s0

    @property
    def observation_space(self):
        return self._observation_space

    ## change end
    ###############################################################
    @property
    def action_space(self):
        return self._env.action_space

    @property
    def action_dim(self):
        return self._env.action_dim

    @property
    def horizon(self):
        return self._env.horizon

    def terminate(self):
        self._env.terminate()

    def render(self):
        self._env.render()
