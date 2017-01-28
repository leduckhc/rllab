from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import ConvNetwork
from rllab.core.serializable import Serializable
from rllab.v_functions.base import VFunction
import lasagne.layers as L
import lasagne.nonlinearities as NL


class ContinuousConvVFunction(VFunction, LasagnePowered, Serializable):
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
                output_dim=1,
                conv_filters=conv_filters,
                conv_filter_sizes=conv_filter_sizes,
                conv_strides=conv_strides,
                conv_pads=conv_pads,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                name="continuous_conv_v_function"
            )

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers

        self._env_spec = env_spec
        self._output_layer = network.output_layer
        self._input_layer = network.input_layer
        self._f_values = ext.compile_function(
            inputs=[network.input_var],
            outputs=L.get_output(network.output_layer, deterministic=True)
        )

        LasagnePowered.__init__(self, [network.output_layer])


    def get_vval(self, observation, **kwargs):
        flat_obs = self.observation_space.flatten(observation)
        return self._f_values([flat_obs])[0]


    def get_vval_sym(self, obs_var, **kwargs):
        flat_obs_var = obs_var.flatten(ndim=2)
        return L.get_output(self._output_layer, flat_obs_var, deterministic=True)
