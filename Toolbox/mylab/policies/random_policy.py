import numpy as np
import tensorflow as tf

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides


class RandomPolicy:
    def __init__(
            self,
            name,
            env_spec,
    ):
        """
        :param env_spec: A spec for the env.
        :return:
        """
        self.name = name
        self.env_spec = env_spec

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        pass

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        action = self.env_spec.action_space.sample()
        return action, {}

    @overrides
    def get_actions(self, observations):
        batch_size = len(observations)
        actions = np.array([self.env_spec.action_space.sample() for i in range(batch_size)])
        return actions, {}

    @property
    @overrides
    def recurrent(self):
        return False

    @property
    def state_info_specs(self):
        return []

