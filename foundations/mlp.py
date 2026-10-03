import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors

        # feed-forward through multiple layers

        # goes through each layer (list of weights)
        n_layers = len(weights)
        for i in range(n_layers):
            pre_act = x @ weights[i] + biases[i]

            # activation: hidden layer ReLU
            if i < len(weights) - 1:
                x = np.maximum(pre_act, 0)
            else:
                x = pre_act

        return np.round(x, 5)