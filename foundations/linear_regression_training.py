import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Computes predictions using input and weights
        #   2. For each weight index j, calcualtes gradient using ground truth
        #   3. Update: weights[j] -= learning_rate * gradient

        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X, initial_weights)

            # Goes through and updates each weight
            for j in range(len(initial_weights)):
                gradient = self.get_derivative(predictions, Y, len(X), X, j)

                # update weights using gradient (direction with less loss)
                initial_weights[j] -= gradient * self.learning_rate

        return np.round(initial_weights, 5)

