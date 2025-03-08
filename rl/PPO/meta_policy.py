"""
actor-critic network for high-level policy
"""

import torch 
import torch.nn as nn
import numpy as np 

from utils.distributions import DiagGaussianNet, CategoricalNet 
from utils.networks import Net 



class MetaPolicy(nn.Module):

    def __init__(self):

        super().__init__()










