import numpy as np
import garage.tf.core.layers as L
import tensorflow as tf
from garage.tf.core.layers_powered import LayersPowered
from garage.tf.core.network import LSTMNetwork
from garage.tf.distributions.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.policies.base import StochasticPolicy

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides


class DeterministicLSTMPolicy(Policy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=32,
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh,
            lstm_layer_cls=L.LSTMLayer,
            use_peepholes=False,
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        with tf.variable_scope(name):
            Serializable.quick_init(self, locals())
            super(GaussianLSTMPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            l_input = L.InputLayer(
                shape=(None, None, input_dim),
                name="input"
            )

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.stack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
                )

            mean_network = LSTMNetwork(
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=action_dim,
                hidden_dim=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                lstm_layer_cls=lstm_layer_cls,
                name="mean_network",
                use_peepholes=use_peepholes,
            )

            self.mean_network = mean_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

            self.f_step_mean = tensor_utils.compile_function(
                [
                    flat_input_var,
                    mean_network.step_prev_state_layer.input_var,
                ],
                L.get_output([
                    mean_network.step_output_layer,
                    mean_network.step_hidden_layer,
                    mean_network.step_cell_layer
                ], {mean_network.step_input_layer: feature_var})
            )


            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            self.prev_actions = None
            self.prev_hiddens = None
            self.prev_cells = None
            
            out_layers = [mean_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))
            self.prev_cells = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.mean_network.hid_init_param.eval()
        self.prev_cells[dones] = self.mean_network.cell_init_param.eval()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
        else:
            all_input = flat_obs
        # probs, hidden_vec, cell_vec = self.f_step_prob(all_input, self.prev_hiddens, self.prev_cells)
        means, hidden_vec, cell_vec = self.f_step_mean(
            all_input, np.hstack([self.prev_hiddens, self.prev_cells]))
        actions =  means
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        self.prev_cells = cell_vec
        agent_info = dict(mean=means)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info

    def get_action_sym(self, obs_var):
        #need to be implemented
        #refer to dist_info_sym of Gaussian policy
        return L.get_output(self.prob_network.output_layer, obs_var)

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim,)),
            ]
        else:
            return []

