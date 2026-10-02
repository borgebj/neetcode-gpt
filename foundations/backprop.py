import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:

    def sigmoid(self, z):
         return 1.0 / (1.0 + np.exp(-z))

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        
        # forward
        z = np.dot(x, w) + b
        y_hat = self.sigmoid(z)

        # loss derivative
        dL_dy = (y_hat - y_true)

        # sigmoid_deriv dy_hat/dz
        dy_hat_dz = y_hat * (1.0 - y_hat)

        # gradient w.r.t z
        dL_dz = dL_dy * dy_hat_dz
        
        # gradient w.r.t weights
        dL_dw = np.round(dL_dz * x, 5)

        # gradient w.r.t bias
        dL_db = round(float(dL_dz), 5)

        return (dL_dw, dL_db)
