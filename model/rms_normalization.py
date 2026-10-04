import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        gamma = np.array(gamma)

        # RMS normalization
        rms = np.sqrt(np.mean(x ** 2) + eps)
        x_hat = x / rms

        # scale without beta
        out = x_hat * gamma

        return np.round(out, 4).tolist()

