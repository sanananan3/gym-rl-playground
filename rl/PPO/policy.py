"""
actor-critic network for low-level policy 
"""

import torch
import torch.nn as nn 
import numpy as np 


from utils.distributions import CategoricalNet, DiagGaussianNet, MultiCategoricalNet
from utils.networks import Net 


EPS = 1e-6  # epsilon value 
OLD_NETWORK = False 



class Policy(nn.Module):

    def __init__(self, observation_space, action_space, hidden_size=256,
                 cnn_layers_params=None, initial_stddev=1.0/3.0, min_stddev = 0.0,
                 stddev_annneal_schedule = None, stddev_transform=torch.nn.functional.softplus):

        super().__init__()

        # using MLP 

        s

        


