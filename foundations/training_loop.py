import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate


        # example: shape = (10, 5)
        # 10 samples, 5 features

        N =  X.shape[0]          # [0] = no. samples   (5 samples)
        W = np.zeros(X.shape[1]) # [1] = no. features  (10 learnable weights)
        b = 0.0

        for _ in range(epochs):

            # forward pass
            y_hat = X @ W + b
            error = (y_hat - y)

            # compute gradients
            # linear regression = no activation = ONLY MSE derivative
            dW = (2 / N) * X.T @ error    # <-- MSE wrt W
            db = (2 / N) * np.sum(error)  # <-- MSE wrt b

            # update weights
            W -= lr * dW
            b -= lr * db

        return (np.round(W, 5), round(b, 5))
