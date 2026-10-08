import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # - contains embedding for each ID

        # token_ids: 1D array of integer token IDs

        # embeddings = [
        #   [0.1], [0.2],   <-- id 0
        #   [0.2], [0.3],   <-- id 1
        # ]
        
        return np.round(embeddings[token_ids], 5)