import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # return np.round(your_answer, 4)

        # subtracts max(z) for numerical stability before computing exp
        shifted = z - np.max(z)

        exp_vals = np.exp(shifted) 
        exp_sum = np.sum(exp_vals)

        return np.round(exp_vals / exp_sum, 4)
