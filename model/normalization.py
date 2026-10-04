import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5

        # normalization
        mean = np.mean(x, axis=0)   # mean / avg
        var = np.var(x, axis=0)     # variance
        eps = 1e-5                  # counters div. by zero
        x_hat = (x - mean) / np.sqrt(var + eps)

        # scales and shirts
        out = gamma * x_hat + beta

        return np.round(out, 5)
