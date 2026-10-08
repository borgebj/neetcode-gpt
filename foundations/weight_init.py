import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # xavier (Glorot) initialization:
        # - keeps activation variance stable across layers
        # - comminly used with sigmoid/tanh activatiins
        torch.manual_seed(0)

        std = (2 / (fan_in + fan_out)) ** 0.5
        weights = torch.randn(fan_out, fan_in) * std

        return weights.round(decimals=4).tolist()


    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Kaiming (He) initialization:
        # - designed for ReLU networks
        # - helps activation variance stable through the layers
        torch.manual_seed(0)

        std = (2 / fan_in) ** 0.5
        weights = torch.randn(fan_out, fan_in) * std

        return weights.round(decimals=4).tolist()


    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        dims = [input_dim] + ([hidden_dim] * num_layers)

        weights = []

        # initializes weights
        for i in range(num_layers):
            fan_in = dims[i]
            fan_out = dims[i+1]

            # 'xavier'
            if init_type == "xavier":
                std = (2 / (fan_in + fan_out)) ** 0.5

            # 'kaiming'
            elif init_type == "kaiming":
                std = (2 / fan_in) ** 0.5
            
            # 'random'
            else:
                std = 1.0
            
            layer_weight = torch.randn(fan_out, fan_in) * std
            weights.append(layer_weight)

        # random start input
        x = torch.randn(1, input_dim)  

        stds = []
        for weight in weights:

            # activation - updates x
            x = torch.relu(x @ weight.T)
            stds.append(round(x.std().item(), 2))

        return [round(s, 2) for s in stds]
