import theano
import numpy as np
import random
import pickle
import multiprocessing as mp
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy
from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal


def _get_scoped_G(G, scope):
    if scope is None:
        return G
    if not hasattr(G, "scopes"):
        G.scopes = dict()
    if scope not in G.scopes:
        G.scopes[scope] = SharedGlobal()
        G.scopes[scope].worker_id = G.worker_id
    return G.scopes[scope]


def _worker_init(G, id):
    G.worker_id = id
    print("init pid_%d", id)


def _worker_populate_task(G, policy, scope=None):
    G = _get_scoped_G(G, scope)
    G.policy = pickle.loads(policy)


def train_thread(G, counter, lock):
    for _ in range(10):
        for _ in range(100):
            for _ in range(random.randrange(100)):
                pass
            with lock:
                counter.value += 1
                if counter.value % 81 == 0:
                    new_i = G.policy.opt_update()
                    print("pid_%d, counter %d, int %d" % (
                        G.worker_id, counter.value, new_i))

class SimplePolicy(Policy, Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())
        self.i = 0
        super(SimplePolicy, self).__init__(None)

    def opt_update(self):
        self.i += 1
        return self.i

    @overrides
    def get_params_internal(self, **tags):  # this gives ALL the vars (not the params values)
        return dict()

    def train(self):
        singleton_pool.initialize(n_parallel=4)
        singleton_pool.run_each(_worker_init,
                                [(id,) for id in range(4)])
        singleton_pool.run_each(_worker_populate_task,
                                [(pickle.dumps(self), None)] * 4)
        manager = mp.Manager()
        counter = manager.Value('i', 0)

        lock = manager.RLock()
        results = singleton_pool.run_each(
            train_thread,
            [(counter, lock)] * 4
        )


if __name__ == "__main__":
    policy = SimplePolicy()
    policy.train()
