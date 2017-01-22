from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import ConvNetwork
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.policies.base import Policy

import numpy as np
import theano.tensor as TT
import lasagne.layers as L
import lasagne.nonlinearities as NL

from rllab.spaces import Discrete, Box


class DeterministicConvPolicy(Policy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            conv_filters,
            conv_filter_sizes,
            conv_strides,
            conv_pads,
            hidden_sizes=[],
            hidden_nonlinearity=NL.rectify,
            output_nonlinearity=NL.linear,
            network=None):
        Serializable.quick_init(self, locals())

        if network is None:
            network = ConvNetwork(
                input_shape=env_spec.observation_space.shape,
                output_dim=env_spec.action_space.flat_dim,
                conv_filters=conv_filters,
                conv_filter_sizes=conv_filter_sizes,
                conv_strides=conv_strides,
                conv_pads=conv_pads,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                name="deterministic_conv_network"
            )

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers

        self._env_spec = env_spec
        self._output_layer = network.output_layer
        self._input_layer = network.input_layer
        self._f_actions = ext.compile_function(
            inputs=[network.input_var],
            outputs=L.get_output(network.output_layer, deterministic=True)
        )

        LasagnePowered.__init__(self, [network.output_layer])
        super(DeterministicConvPolicy, self).__init__(env_spec)

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        values = self._f_actions([flat_obs])[0]
        if isinstance(self._env_spec.action_space, Discrete):
            # actually index of an action
            action = np.argmax(values)
        elif isinstance(self._env_spec.action_space, Box):
            action = values
        else:
            raise NotImplementedError
        return action, dict(action_values=values)

    def get_actions(self, observations):
        flat_obs= self.observation_space.flatten_n(observations)
        values = self._f_actions(flat_obs)
        if isinstance(self._env_spec.action_space, Discrete):
            actions = list(map(np.argmax, values))
        elif isinstance(self._env_spec.action_space, Box):
            actions = values
        else:
            raise NotImplementedError
        return actions, dict(action_values=values)

    def get_action_sym(self, obs_var):
        flat_obs_var = obs_var.flatten(ndim=2)
        return L.get_output(self._output_layer, flat_obs_var, deterministic=True)
        # values_var = TT.reshape(values_var, (-1,))
        # if isinstance(self._env_spec.action_space, Discrete):
        #     return TT.argmax(values_var), dict(action_values=values_var)
        # elif isinstance(self._env_spec.action_space, Box):
        #     return values_var, dict(action_values=values_var)
        # else:
        #     raise NotImplementedError

