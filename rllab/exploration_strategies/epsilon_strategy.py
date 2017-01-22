from rllab.core.serializable import Serializable
from rllab.exploration_strategies.base import ExplorationStrategy
from numpy.random import random

from rllab.misc.overrides import overrides


class EpsilonGreedyStrategy(ExplorationStrategy, Serializable):
    """
    Epsilon greedy exploration strategy.
    """

    def __init__(
            self,
            env_spec,
            eps_start=1.0,
            eps_final=0.1,
            eps_itr_start=10000,
            eps_itr_final=1000000):
        """
        Epsilon-greedy exploration strategy.

        Parameters
        ----------
        env_spec :
        eps_start :
        eps_final :
        eps_itr_start :
        eps_itr_final :
        """
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_itr_final = eps_itr_final
        self.eps_itr_start = eps_itr_start

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        if t < self.eps_itr_start:
            eps = self.eps_start
        elif t > self.eps_itr_final:
            eps = self.eps_final
        else:
            # linearly anneal
            eps = self.eps_start + \
                  (self.eps_final - self.eps_start) * (t - self.eps_itr_start) * \
                  1. / (self.eps_itr_final - self.eps_itr_start)
        if random() < eps:
            return self.env_spec.action_space.sample()
        action, _ = policy.get_action(observation)
        return action
