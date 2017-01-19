import pickle
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import pyprind
import random
import theano.tensor as T
from experiments.dqn.atari_env_wrapper import AtariEnvWrapper
from lasagne.updates import rmsprop

import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.algos.base import RLAlgorithm
from rllab.algos.util import ReplayPool
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import ConvNetwork, visualize_conv_weight, visualize_conv_activation
from rllab.core.serializable import Serializable
from rllab.envs.env_spec import EnvSpec
from rllab.misc import ext
from rllab.misc import special
from rllab.policies.base import Policy
from rllab.q_functions.base import QFunction
from rllab.sampler import parallel_sampler
from rllab.spaces.discrete import Discrete


class DeepQPolicy(Policy, QFunction, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            conv_filters=(16, 32),
            conv_filter_sizes=(8, 4),
            conv_strides=(4, 2),
            conv_pads=(0, 0),
            hidden_sizes=[256],
            hidden_nonlinearity=NL.rectify,
            output_nonlinearity=NL.linear,
    ):
        """
        The policy consists of convolution layers followed by dense layers. See the original paper.

        :param env_spec: A spec for the mdp.
        :type env_spec: EnvSpec
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for all hidden layers

        :return: Deep Q-function Network

        Note
        ----
        The Feedforward and Backward pass uses a lot of cpu power.

        [1] V.Mnih
        """
        Serializable.quick_init(self, locals())
        
        assert isinstance(env_spec.action_space, Discrete)
        
        self._env_spec = env_spec
        
        network = ConvNetwork(
            input_shape=(4, 84, 84),
            output_dim=env_spec.action_space.n,
            conv_filters=conv_filters,
            conv_filter_sizes=conv_filter_sizes,
            conv_strides=conv_strides,
            conv_pads=conv_pads,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )
        
        self._l_qval = network.output_layer
        self._l_obs = network.input_layer
        self._f_qval = ext.compile_function(
            [network.input_layer.input_var],
            L.get_output(network.output_layer)
        )
        self._iseval = False
        
        super(DeepQPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [network.output_layer])
    
    def get_qval(self, observation):
        flat_obs = self._env_spec.observation_space.flatten_n(observation)
        return self._f_qval(flat_obs)
        
    def get_qval_sym(self, obs_var):
        """
        Parameters
        ----------
            obs_var : theano.tensor
                Symbolic observation variable. Usually created by TT.vector('name')
        
        Returns
        -------
            symbolic expression
                Symbolic value of Q-value
        """
        return L.get_output(self._l_qval, obs_var)
    
    def get_action(self, observation):
        """
        Get action and its associate Q-value

        :param observation: obs
        :return: The return value is an action)).
        """
        qval = self.get_qval([observation])[0]
        action = np.argmax(qval)
        return action, dict(qval=qval)
    
    def get_trainable_params(self):
        return L.get_all_params(self._l_qval, traiable=True)
    
    def reset(self):
        pass
    
    def visualize_conv_weight(self, epoch):
        conv_layer = L.get_all_layers(self._l_qval)[2]
        visualize_conv_weight(conv_layer, epoch)
        
    def visualize_conv_activation(self, obs, epoch):
        conv_layer = L.get_all_layers(self._l_qval)[2]
        obs = self._env_spec.observation_space.flatten_n(obs)
        visualize_conv_activation(conv_layer, obs, epoch)
    
def epsilon_greedy(itr):
    eps_start, eps_final = 1.0, 0.1
    eps_itr_start, eps_itr_final = 1e4, 1e6

    if itr < eps_itr_start:
        eps = eps_start
    elif itr > eps_itr_final:
        eps = eps_final
    else:
        eps = eps_start + \
              (eps_final - eps_start) * (itr - eps_itr_start) * \
              1. / (eps_itr_final - eps_itr_start)

    return random.random() < eps

class DQN(RLAlgorithm, Serializable):
    def __init__(
            self,
            env,
            scope=None,
            agent_history_length=4,
            n_epochs=200,
            batch_size=32,
            discount=0.99,
            replay_memory_size=1000000,
            min_replay_memory_size=50000,
            epoch_length=50000,
            max_path_length=10000,
            target_network_update_frequency=10000,
            eval_max_samples=1000,
            eval_max_path_length=10000,
            plot=False,
    ):
        """Returns Deep Q-Network algorithm.
        
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
        max_path_length : int
            Each rollout/path has max lengh. After this we will not restart epoch.
        target_network_update_frequency : int
        plot : bool
        """

        assert replay_memory_size > min_replay_memory_size

        Serializable.quick_init(self, locals())

        self.env = AtariEnvWrapper(env, agent_history_length, 84, 84)
        self.env_spec = self.env.spec
        self.scope = scope
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        self.discount = discount
        self.replay_memory_size = replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.target_network_update_frequency = target_network_update_frequency
        self.max_path_length = max_path_length
        self.agent_history_length = agent_history_length
        self.eval_max_samples = eval_max_samples
        self.eval_max_path_length = eval_max_path_length
        self.plot = plot
        
        self.policy = DeepQPolicy(env_spec=self.env_spec)
        self.policy_stale = pickle.loads(pickle.dumps(self.policy))
        
        self.qf_loss_averages = []
        self.qs_averages = []
        self.ys_averages = []
    
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
            concat_length=1, # env wrapper se o to postara o concatenaci 4 framu
        )

        self.start_worker()
        self.init_opt()

        itr = 0
        update_freq_itr = 0
        path_length = 0
        path_return = 0
        terminal = False

        obs = self.env.reset()

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch %d | ' % epoch)
            logger.log('Training started')
            for step in pyprind.prog_bar(range(self.epoch_length)):
                if len(replay_memory) <= 4 or epsilon_greedy(itr):
                    action = self.env.action_space.sample()
                else:
                    n_obs = replay_memory.last_concat_state()
                    action, _ = self.policy.get_action(n_obs)
                    
                # for _ in range(4): # skip frames not needed because gym makes it for us
                next_obs, reward, terminal, env_info = self.env.step(action)
                replay_memory.add_sample(obs, action, np.clip(reward, -1.0, +1.0), terminal)
                obs = next_obs if not terminal else self.env.reset()

                # Training/learning phase starts after some steps.
                # Read the paper for details.
                if len(replay_memory) >= self.min_replay_memory_size:
                    memory_samples = replay_memory.random_batch(self.batch_size)
                    self.do_training(itr, memory_samples)
                    update_freq_itr += 1
                    if update_freq_itr % self.target_network_update_frequency == 0:
                        self.policy_stale.set_param_values(self.policy.get_param_values())

                itr += 1

            logger.log('Training finished')
            if len(replay_memory) >= self.min_replay_memory_size:
                self.evaluate(epoch, replay_memory)
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)

            # params = self.get_epoch_snapshot(epoch)
            # logger.save_itr_params(epoch, params)
            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.update_plot()
                if self.pause_for_plot:
                    input('Plotting evaluation run: Press Enter to continue...')
        self.shutdown_worker()

    def do_training(self, itr, samples):
        """Update the policy a save the values for logging.

        Parameters
        ----------
        itr : int
            Number of frames/steps
        samples : dict
            Contains keys (observations, actions, rewards, terminals,
            next_observations, ...). Usually random batch from ReplayPool
        """
        
        s0, a0, r0, t0, s1 = ext.extract(
            samples,
            'observations', 'actions', 'rewards', 'terminals', 'next_observations'
        )
        n_samples = len(s0)
        
        qval = self.policy.get_qval(s0)
        qf = [qval[i, int(a0[i])] for i in range(n_samples)]
        qval_stale = self.policy_stale.get_qval(s1)
        target = r0 + self.discount * (1. - t0) * np.max(qval_stale, axis=1)
        
        # we should append qf_loss to some array to track loss while training. But not now
        # logger.log('shape qf=(%s)  target=(%s)' % (qf.shape, target.shape))
        qf_loss = self.f_train(qf, target)
        
        # store the values for logging
        self.qf_loss_averages.append(qf_loss)
        self.qs_averages.append(qf)
        self.ys_averages.append(target)
    
    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano, this may
        include declaring all the variables and compiling functions.
        """
        
        # building network
        qval_var = T.vector('qval', dtype='floatX')
        target_var = T.vector('target', dtype='floatX')
        
        loss_var = T.mean(T.square(target_var - qval_var))
        params = self.policy.get_trainable_params()
        updates = rmsprop(loss_var, params,
            learning_rate=0.0002, rho=0.95, epsilon=1e-6)
        
        self.f_train = ext.compile_function(
            inputs=[qval_var, target_var],
            outputs=loss_var,
            updates=updates,
            name='f_train',
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
