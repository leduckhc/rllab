from __future__ import print_function

import pickle

import numpy as np
import pyprind
import theano
import theano.tensor as TT
import lasagne

import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.algos.base import RLAlgorithm
from rllab.algos.util import ReplayPool
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import special
from rllab.sampler import parallel_sampler
from functools import partial


def parse_update_method(update_method, **kwargs):
    if update_method == 'adam':
        return partial(lasagne.updates.adam, **ext.compact(kwargs))
    elif update_method == 'rmsprop':
        return partial(lasagne.updates.rmsprop, **ext.compact(kwargs))
    else:
        raise NotImplementedError


class DQN(RLAlgorithm, Serializable):
    """
    Deep Q-Network algorithm [1]_ [2]_

    Notes
    -----
    * Gym takes care of frame skipping.
    * Agent history concatenation not needed due to class :class:`rllab.envs.SlidingMemEnv`.
    * Class :class:`rllab.envs.PreproocessEnv` is responsible for resizing image and transform it into grayscale.

    .. [1] Mnih, Volodomyr, et al. "Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree
        Search Planning." NIPS. 2015.
    .. [2] Mnih, Volodomyr, et al. "Human-level control through deep reinforcement
        learning." Nature. 2015.
    """

    def __init__(
            self,
            env,
            policy,
            es,
            scope=None,
            n_epochs=500,
            epoch_length=50000,
            batch_size=32,
            discount=0.99,
            replay_memory_size=1000000,
            min_replay_memory_size=50000,
            target_network_update_frequency=10000,
            agent_history_length=4,
            resized_shape=(84,84),
            eval_max_samples=100000,
            eval_max_path_length=2000,
            update_method='rmsprop',
            update_method_kwargs=None,
            plot=False,
            max_succesive_noop=30,
    ):
        """Deep Q-Network algorithm [1]_ [2]_
        
        Parameters
        ----------
        env : gym_env.AtariGymEnv
            Atari environment
        scope : string
            Scope for multiple variants
        n_epochs : int
            Number of epochs
        batch_size : int
            In each step we optimize with batch size
        discount : float
        replay_memory_size : int
        min_replay_memory_size : int
            Iteration at which we start optimizing using replay memory
        epoch_length : int
            Number of steps in each epoch. If some state is terminal we won't
            reset the whole epoch but rather continue
        resized_shape : tuple
        eval_max_path_length : int
            Each rollout/path has max lengh. After this we will not restart epoch.
        target_network_update_frequency : int
<<<<<<< 33d4b7a59e6eda50a33af9b4bbf6046776b88aaa
=======
        soft_target_tau : float
            parameters of cur_policy == (1-tau)*cur_policy + tau*old_policy
>>>>>>> small fix dqn_demo
        update_method : str
            'rmsprop' or 'adam'
        update_method_kwargs : dict
            kwargs for update method
        plot : bool

        Notes
        -----
        * Gym takes care of frame skipping.
        * Agent history concatenation not needed due to class :class:`rllab.envs.SlidingMemEnv`.
        * Class :class:`rllab.envs.PreproocessEnv` is responsible for resizing image and transform it into grayscale.

        .. [1] Mnih, Volodomyr, et al. "Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree
            Search Planning." NIPS. 2015.
        .. [2] Mnih, Volodomyr, et al. "Human-level control through deep reinforcement
            learning." Nature. 2015.

        """

        assert replay_memory_size >= min_replay_memory_size
        assert eval_max_samples >= eval_max_path_length

        Serializable.quick_init(self, locals())

        self.env = env
        self.policy = policy
        self.es = es
        self.scope = scope

        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.discount = discount
        self.replay_memory_size = replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.target_network_update_frequency = target_network_update_frequency
        self.agent_history_length = agent_history_length
        self.resized_shape = resized_shape
        self.eval_max_samples = eval_max_samples
        self.eval_max_path_length = eval_max_path_length
        self.plot = plot
        if update_method_kwargs is None:
            update_method_kwargs = dict(
                learning_rate=0.00025 , rho=0.95, epsilon=1e-2)
        self.update_method = parse_update_method(update_method, **update_method_kwargs)
        self.max_succesive_noop = max_succesive_noop

        self.qf_loss_averages = []
        self.qs_averages = []
        self.ys_averages = []
        self.es_path_returns = []
        self.es_path_length = []

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        parallel_sampler.terminate_task()

    def train(self):
        replay_memory = ReplayPool(
            observation_shape=self.resized_shape,
            action_dim=1,
            action_dtype='int8',
            max_steps=self.replay_memory_size,
            concat_observations=True,
            concat_length=4,
        )

        self.start_worker()
        self.init_opt()

        itr = 0
        update_freq_itr = 0
        path_length = 0
        path_return = 0
        terminal = False
        succesive_noop = 0

        obs = self.env.reset()

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch %d | ' % epoch)
            logger.log('Training started')
            for step in pyprind.prog_bar(range(self.epoch_length)):

                if terminal:
                    obs = self.env.reset()
                    self.es.reset()
                    self.policy.reset()
                    self.es_path_returns.append(path_return)
                    self.es_path_length.append(path_length)
                    path_length = 0
                    path_return = 0

                # Execute policy
                action = self.es.get_action(itr, obs, policy=self.policy)
                if action == 0: succesive_noop += 1
                else: succesive_noop = 0

                if succesive_noop == self.max_succesive_noop:
                    # exclude noop operations
                    succesive_noop = 0
                    action = np.random.randint(1, self.env.action_space.n)

                next_obs, reward, terminal, env_info = self.env.step(action)
                path_length += 1
                path_return += reward
                replay_memory.add_sample(obs[-1], action, np.clip(reward, -1.0, +1.0), terminal)

                # Training/learning phase does not start directly.
                # It lets the algo to explore the env. Read the paper for details.
                if len(replay_memory) >= self.min_replay_memory_size:
                    batch = replay_memory.random_batch(self.batch_size)
                    self.do_training(itr, batch)
                    update_freq_itr += 1
                    if update_freq_itr % self.target_network_update_frequency == 0:
                        self.opt_info['target_policy'].set_param_values(self.policy.get_param_values())

                obs = next_obs
                itr += 1

            logger.log('Training finished')
            if len(replay_memory) >= self.min_replay_memory_size:
                self.evaluate(epoch, replay_memory)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    input('Plotting evaluation run: Press Enter to continue...')

        self.shutdown_worker()

    def do_training(self, itr, samples):
        """Update the policy and save the losses and vqalues for logging.

        Parameters
        ----------
        itr : int
            Number of frames/steps
        samples : dict
            Contains keys (observations, actions, rewards, terminals,
            next_observations, ...). Usually random batch from ReplayPool
        """

        observations, actions, rewards, terminals, next_observations = ext.extract(
            samples,
            'observations', 'actions', 'rewards', 'terminals', 'next_observations'
        )

        f_train_policy = self.opt_info['f_train_policy']
        target_policy = self.opt_info['target_policy']

        # target values ys. Each row is a sample so we need axis=1
        _, action_info = target_policy.get_actions(next_observations)
        target_qvalues = np.max(action_info['action_values'], axis=1)
        targets = rewards +  self.discount * (1. - terminals) * target_qvalues
        flat_observations = self.env.observation_space.flatten_n(observations)

        # f_train_policy updates the parameters of self.policy
        qf_loss, qvalues = f_train_policy(flat_observations, actions, targets)

        # store the values for logging
        self.qf_loss_averages.append(qf_loss)
        self.qs_averages.append(qvalues)
        self.ys_averages.append(targets)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano, this may
        include declaring all the variables and compiling functions.
        """
        # specify target policy
        target_policy = pickle.loads(pickle.dumps(self.policy))
        target_var = TT.vector('target', dtype=theano.config.floatX)

        # building network
        obs_var = self.env.observation_space.new_tensor_variable('obs', extra_dims=1)
        action_var = self.env.action_space.new_tensor_variable('action', extra_dims=1)
        qval_var_all = self.policy.get_action_sym(obs_var)
        qval_var = qval_var_all[TT.arange(qval_var_all.shape[0]), action_var]

        loss_var = TT.mean(TT.square(target_var - qval_var))
        params = self.policy.get_params(trainable=True)
        updates = self.update_method(loss_var, params)

        # debugging functions
        # also uncomment mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs)
        # def inspect_inputs(i, node, fn):
        #     print(i, node, "input(s) shape(s):", [input[0].shape for input in fn.inputs],end='')
        # def inspect_outputs(i, node, fn):
        #     print(" output(s) shape(s):", [output[0].shape for output in fn.outputs])

        f_train_policy = ext.compile_function(
            inputs=[obs_var, action_var, target_var],
            outputs=[loss_var, qval_var],
            updates=updates,
            name='f_train_policy',
            # mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs)
        )

        self.opt_info = dict(
            f_train_policy = f_train_policy,
            target_policy = target_policy
        )

    def evaluate(self, epoch, pool):
        logger.log('Collecting samples for evaluation')

        paths = parallel_sampler.sample_paths(
            policy_params=self.policy.get_param_values(),
            max_samples=self.eval_max_samples,
            max_path_length=self.eval_max_path_length,
        )

        average_discounted_return = np.mean(
            [special.discount_return(path['rewards'], self.discount) for path in paths]
        )
        returns = [sum(path['rewards']) for path in paths]

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('AverageDiscountedReturn', average_discounted_return)
        logger.record_tabular('StdReturn', np.std(returns))
        logger.record_tabular('MaxReturn', np.max(returns))
        logger.record_tabular('MinReturn', np.min(returns))

        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
            logger.record_tabular('AverageEsPathLength',
                                  np.mean(self.es_path_length))

        logger.record_tabular('AverageQLoss', np.mean(self.qf_loss_averages))

        all_qs = np.concatenate(self.qs_averages)
        all_ys = np.concatenate(self.ys_averages)

        logger.record_tabular('AverageQ', np.mean(all_qs))
        logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
        logger.record_tabular('AverageY', np.mean(all_ys))
        logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))

        self.qf_loss_averages = []
        self.qs_averages = []
        self.ys_averages = []

        self.es_path_length=[]
        self.es_path_returns=[]

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            env=self.env,
            policy=self.policy,
            target_policy=self.opt_info['target_policy'],
        )
