import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> returns (n,) predictions
        
        # prediction done using matrix multiplication
        prediction = X @ weights 
        return np.round(prediction, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Computes mean squared error between predictions and ground truth

        # 1. calculates error (y - p)
        # squares error ^2 
        # sums errors
        # averages
        mse = np.mean((ground_truth - model_prediction) ** 2)
        return np.round(mse, 5)
