import multiprocessing
import joblib.pickle as pickle
import lasagne
import theano.tensor as TT
import numpy as np

from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.plotter import plotter
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool


def optimize_policy(sample_data):
    inputs = ext.extract(
        sample_data,
        "observations", "actions", "advantages"
    )


def _train_worker(G, g_counter, env, opt_info, target_net, t_max, discount, lock):
    target_policy = target_net["target_policy"]
    target_vfunc = target_net["target_vfunc"]
    policy = pickle.loads(pickle.loads(target_policy))
    baseline = pickle.loads(pickle.loads(target_vfunc))

    t_local = 1
    obs = env.reset()
    observations = []
    actions = []
    rewards = []
    paths = []
    while True:
        t_start = t_local
        done = False
        while True:
            action, _ = policy.get_action(obs)
            next_obs, reward, done, info = obs.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs if not done else env.reset()
            t_local += 1
            with lock:
                g_counter.value += 1
            if done or t_local - t_start == t_max:
                break

        # make it not expanding
        observations = observations[-t_max:]
        actions = actions[-t_max:]
        rewards = rewards[-t_max:]

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards)
        )

        path_baseline = baseline.predict(path)
        advantages = []
        returns = []
        return_so_far = 0 if done else path_baseline[-1:]
        for t in range(len(rewards) - 1, -1, -1):
            return_so_far = rewards[t] + discount * return_so_far
            returns.append(return_so_far)
            advantage = return_so_far - path_baseline[t]
            advantages.append(advantage)
        # The advantages are stored backwards in time, so we need to revert it
        advantages = np.array(advantages[::-1])
        # The returns are stored backwards in time, so we need to revert it
        returns = returns[::-1]
        # normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        path["advantages"] = advantages
        path["returns"] = returns
        paths.apend(path)

        optimize_policy(path)
        baseline.fit(path)











class A3C(RLAlgorithm, Serializable):
    def __init__(
            self,
            env,
            policy,
            vfunc,
            es,
            scope=None,
            n_epochs=200,
            t_max=5,
            vfunc_weight_decay=1e-3,
            vfunc_update_method=lasagne.updates.rmsprop,
            policy_weight_decay=1e-3,
            policy_update_method=lasagne.updates.rmsprop,
            scale_reward=0.1):
        Serializable.quick_init(locals_=locals())

        self.env = env
        self.policy = policy
        self.vfunc = vfunc
        self.es = es
        self.scope = scope
        self.n_epochs = n_epochs
        self.t_max = t_max
        self.vfunc_weight_decay = vfunc_weight_decay
        self.policy_weight_decay = policy_weight_decay
        self.vfunc_update_method = vfunc_update_method
        self.policy_update_method = policy_update_method

        self.scale_reward = scale_reward

        self.opt_info = None
        self.target_net = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy, self.scope)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def train(self):
        self.start_worker()

        manager = multiprocessing.Manager()
        g_counter = manager.Value('T', 0)  # global counter
        # g_policy_param = manager.Value('policy_param', policy_param)
        # g_vfunc_param = manager.Value('vfunc_param', vfunc_param)
        lock = manager.RLock()

        for epoch in range(self.n_epochs):

            results = singleton_pool.run_each(
                _train_worker,
                [(g_counter, self.env, self.opt_info, self.target_net, self.t_max, self.discount, lock)] *
                singleton_pool.n_parallel
            )

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
        )

        self.target_net = dict(
            target_policy=policy,
            target_vfunc=vfunc
        )


    def terminate_task(self):
        # waits till all the processes finished
        parallel_sampler.terminate_task(self.scope)
