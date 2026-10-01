import numpy as np
from numpy.typing import NDArray


class Solution:

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"

        # pre-activation
        z = np.dot(x, w) + b

        # activation
        if activation == "relu":
            a = max(0.0, z)
        
        elif activation == "sigmoid":
            a = 1.0 / (1.0 + np.exp(-z))
        
        else:
            a = z

        return np.round(a, 5)