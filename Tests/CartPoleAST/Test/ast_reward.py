
# useful packages for math and debugging
import numpy as np
import pdb

# Define the class, inherit from the base
class ASTReward(object):
    def __init__(self):
        pass

    def give_reward(self, action, **kwargs):
        """
        returns the reward for a given time step
        Input
        -----
        kwargs : accepts relevant info for computing the reward
        Outputs
        -------
        (observation, reward, done, info)
        reward [Float] : amount of reward due to the previous action
        """

        raise NotImplementedError