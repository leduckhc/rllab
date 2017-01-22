import pickle
import numpy as np
import pyprind
import theano.tensor as T
from lasagne.updates import rmsprop

import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.algos.base import RLAlgorithm
from rllab.algos.util import ReplayPool
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.envs.env_spec import EnvSpec
from rllab.misc import ext
from rllab.misc import special
from rllab.policies.base import Policy
from rllab.sampler import parallel_sampler


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
            n_epochs=200,
            epoch_length=50000,
            batch_size=32,
            discount=0.99,
            replay_memory_size=200000,
            min_replay_memory_size=10000,
            target_network_update_frequency=10000,
            agent_history_length=4,
            eval_max_samples=50000,
            eval_max_path_length=1000,
            plot=False,
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
        eval_max_path_length : int
            Each rollout/path has max lengh. After this we will not restart epoch.
        target_network_update_frequency : int
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
        self.eval_max_samples = eval_max_samples
        self.eval_max_path_length = eval_max_path_length
        self.plot = plot

        self.qf_loss_averages = []
        self.qs_averages = []
        self.ys_averages = []

        self.opt_info = None

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        parallel_sampler.terminate_task()

    def train(self):
        replay_memory = ReplayPool(
            observation_shape=self.env.observation_space.shape,
            action_dim=1,
            action_dtype='int8',
            max_steps=self.replay_memory_size,
            concat_observations=False,
            # env wrapper takes care of agent_history_length
            concat_length=1,
        )

        self.start_worker()
        self.init_opt()

        itr = 0
        update_freq_itr = 0
        path_length = 0
        path_return = 0
        terminal = False

        sample_policy = pickle.loads(pickle.dumps(self.policy))
        obs = self.env.reset()

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch %d | ' % epoch)
            logger.log('Training started')
            for step in pyprind.prog_bar(range(self.epoch_length)):

                if terminal:
                    obs = self.env.reset()
                    self.es.reset()
                    sample_policy.reset()
                    self.es_path_returns.append(path_return)
                    path_length = 0
                    path_return = 0

                # Execute policy
                action = self.es.get_action(itr, obs, policy=sample_policy)
                next_obs, reward, terminal, env_info = self.env.step(action)
                path_length += 1
                path_return += reward
                replay_memory.add_sample(obs, action, np.clip(reward, -1.0, +1.0), terminal)

                # Training/learning phase starts after some steps.
                # Read the paper for details.
                if len(replay_memory) >= self.min_replay_memory_size:
                    batch = replay_memory.random_batch(self.batch_size)
                    self.do_training(itr, batch)
                    update_freq_itr += 1
                    if update_freq_itr % self.target_network_update_frequency == 0:
                        sample_policy.set_param_values(self.policy.get_param_values())

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
        """Update the policy and save the values for logging.

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
        batch_size = len(observations)

        f_train_policy = self.opt_info['f_train_policy']
        target_policy = self.opt_info['target_policy']

        # target values ys. Each row is a sample so we need axis=1
        _, action_info = target_policy.get_actions(next_observations)
        target_qvalues = action_info['values']
        targets = rewards + (1. - terminals) * self.discount * np.max(target_qvalues, axis=1)

        #TODO DEBUG
        # for now we don't care about action, but rather q-values
        # we implemted qvalues little verbose so that it is easy to debug, actually we could do
        # qvalues = [batch_qvalues[i, int(actions[ind])] for i in range batch_size]
        _, action_info = self.policy.get_actions(observations)
        batch_qvalues = action_info["values"]
        qvalues = []
        for ind in range(batch_size):
            # actions.dtype is floatX
            action_ind = int(actions[ind])
            qvalue = batch_qvalues[ind, action_ind]
            qvalues.append(qvalue)

        qf_loss = f_train_policy(qvalues, targets)

        target_policy.set_param_values(
            target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
            self.policy.get_param_values() * self.soft_target_tau)

        # store the values for logging
        self.qf_loss_averages.append(qf_loss)
        self.qs_averages.append(qvalue)
        self.ys_averages.append(targets)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano, this may
        include declaring all the variables and compiling functions.
        """

        # building network
        qval_var = T.vector('qval', dtype='floatX')
        target_var = T.vector('target', dtype='floatX')

        loss_var = T.mean(T.square(target_var - qval_var))
        params = self.policy.get_params(trainable=True)
        updates = rmsprop(loss_var, params,
                          learning_rate=0.0002, rho=0.95, epsilon=1e-6)

        f_train_policy = ext.compile_function(
            inputs=[qval_var, target_var],
            outputs=loss_var,
            updates=updates,
            name='f_train_policy',
        )

        target_policy = pickle.loads(pickle.dumps(self.policy))

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

        logger.record_tabular('AverageQLoss', np.mean(self.qf_loss_averages))

        if len(self.qs_averages) > 0 and len(self.ys_averages) > 0:
            all_qs = np.concatenate(self.qs_averages)
            all_ys = np.concatenate(self.ys_averages)

            logger.record_tabular('AverageQ', np.mean(all_qs))
            logger.record_tabular('AverageAbsQ', np.mean(np.abs(all_qs)))
            logger.record_tabular('AverageY', np.mean(all_ys))
            logger.record_tabular('AverageAbsY', np.mean(np.abs(all_ys)))

        self.qf_loss_averages = []
        self.qs_averages = []
        self.ys_averages = []

        # logger.log("Generating plots of networks")
        # self.policy.visualize_conv_weight(epoch)
        # best_return_ind = np.argmax(returns)
        # some_obs_length = len(paths[best_return_ind]['observations'])
        # some_obs_ind = np.linspace(0, some_obs_length, 10, endpoint=False, dtype=np.intp)
        # some_obs = paths[best_return_ind]['observations'][some_obs_ind]
        # self.policy.visualize_conv_activation(some_obs, epoch)

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            env=self.env,
            policy=self.policy,
            target_policy=self.policy_stale,
        )
