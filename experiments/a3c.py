import multiprocessing
import joblib.pickle as pickle
import lasagne
import theano.tensor as TT
import numpy as np
import time

from experiments.a3c_lib import *
from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.plotter import plotter
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool, ProgBarCounter


class A3C(RLAlgorithm, Serializable):
    def __init__(
            self,
            env,
            policy,
            vfunc,
            es,
            scope=None,
            n_epochs=200,
            epoch_length=10000,
            t_max=5,
            vfunc_weight_decay=1e-3,
            vfunc_update_method=lasagne.updates.rmsprop,
            policy_weight_decay=1e-3,
            policy_update_method=lasagne.updates.rmsprop,
            scale_reward=0.1):

        Serializable.quick_init(self, locals())

        self.env = env
        self.policy = policy
        self.vfunc = vfunc
        self.es = es
        self.scope = scope
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.t_max = t_max
        self.vfunc_weight_decay = vfunc_weight_decay
        self.policy_weight_decay = policy_weight_decay
        self.vfunc_update_method = vfunc_update_method
        self.policy_update_method = policy_update_method
        self.scale_reward = scale_reward

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy, self.scope)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def train(self):
        self.start_worker()
        self.init_opt()

        manager = multiprocessing.Manager()
        lock = manager.RLock()
        g_counter = manager.Value('counter', 0)
        g_opt_info = manager.Value('opt_info', self.opt_info)

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch %d | ' % epoch)
            logger.log('Training started')

            results = singleton_pool.run_each(
                train_worker,
                [(g_counter, g_opt_info, self.t_max, self.discount, lock, self.scope)] *
                singleton_pool.n_parallel
            )

            threshold = self.epoch_length
            pbar = ProgBarCounter(threshold)
            last_value = 0
            while True:
                time.sleep(0.1)
                with lock:
                    if g_counter.value >= threshold:
                        logger.log('Training finished')
                        pbar.stop()
                        g_counter.value = 0
                        logger.log('Evaluating ...')
                        self.evaluate(g_opt_info.value)
                        logger.dump_tabular(with_prefix=False)
                        logger.pop_prefix()
                        break
                    pbar.inc(g_counter.value - last_value)
                    last_value = g_counter.value

        self.terminate_task()

    def init_opt(self):
        # First, create "target" policy and Q functions
        policy = pickle.loads(pickle.dumps(self.policy))
        vfunc = pickle.loads(pickle.dumps(self.vfunc))

        # Building computation graph we need some symbolic variables.
        # observations is batch of obs and similarly actions is batch of single actions
        observations_var = self.env.observation_space.new_tensor_variable(
            'obs',
            # It should have 1 extra dimension since we want to represent a list of observations
            extra_dims=1,
        )
        actions_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantages_var = TT.vector('advantages')

        # ========== VALUE FUNC ===============================================
        vfunc_weight_decay_term = 0.5 * self.vfunc_weight_decay * \
                                  sum([TT.sum(TT.square(param)) for param in
                                       self.vfunc.get_params(regularizable=True)])

        # Symbolic values of value function parameters (eg. weights and biases of MLP)
        vval = self.vfunc.get_vval_sym(observations_var)
        vfunc_loss = TT.mean(TT.square(advantages_var))
        vfunc_reg_loss = vfunc_loss + vfunc_weight_decay_term

        # ========== POLICY ===================================================
        policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([TT.sum(TT.square(param))
                                        for param in self.policy.get_params(regularizable=True)])

        # Symbolic values of distribution parameters, ie. gaussian policy it is mean va std
        dist_info_vars = self.policy.dist_info_sym(observations_var)
        dist = self.policy.distribution
        policy_surr = - TT.mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * advantages_var)
        policy_reg_surr = policy_surr + policy_weight_decay_term

        # ========== DEFINE TRAINING FUNCTIONS ================================
        vfunc_updates = self.vfunc_update_method(
            vfunc_reg_loss, self.vfunc.get_params(trainable=True))
        policy_updates = self.policy_update_method(
            policy_reg_surr, self.policy.get_params(trainable=True))

        f_train_vfunc = ext.compile_function(
            inputs=[observations_var, advantages_var],
            outputs=[vfunc_loss, vval],
            updates=vfunc_updates
        )

        f_train_policy = ext.compile_function(
            inputs=[observations_var, actions_var, advantages_var],
            outputs=policy_surr,
            updates=policy_updates
        )

        self.opt_info = dict(
            f_train_vfunc=f_train_vfunc,
            f_train_policy=f_train_policy,
            target_policy=policy,
            target_vfunc=vfunc
        )

    def evaluate(self, value):
        pass

    def terminate_task(self):
        parallel_sampler.terminate_task(self.scope)
