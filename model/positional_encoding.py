import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # seq_len: no. tokens (rows) / no. positions
        # d_model: no. dimensions in pos vector (columns)

        # positional vectors in shape (seq_len, d_model) -> e.g. (2, 4) -> 2x4 matrix
        # PE = [0   0   0   0]
        #      [0   0   0   0]
        PE = np.zeros((seq_len, d_model))

        # positions for 2 tokens
        # [0]  -  first tokens
        # [1]  -  second token
        positions = np.arange(seq_len).reshape(-1, 1)

        # pairs from 0 to n-dimensions
        pairs = np.arange(0, d_model, 2)

        # frequency denominators
        div_term = 10000 ** (pairs / d_model)

        # even columns get sine   -  0,2,4,6,...
        PE[:, 0::2] = np.sin(positions / div_term)

        # odd columns get cosine  -  1,3,5,7,...
        PE[:, 1::2] = np.cos(positions / div_term)

        return np.round(PE, 5)
