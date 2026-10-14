import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        # Self-attention layer
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)   # K (Key)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False) # Q (Query)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False) # V (value)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)

        # swaps second and third dimension:   
        #   (2,2,4)@(2,2,4)    doesnt work
        #   (2,2,4)@(2,4,2)^T  works!
        K_T = torch.transpose(K, 1, 2)  
        batch_size, no_tokens, n_dim = K.shape
        attention_scores = (Q @ K_T) / (n_dim ** 0.5)

        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')

        # mask prevents looking at future tokens
        mask = torch.tril(torch.ones(no_tokens, no_tokens))
        attention_scores = attention_scores.masked_fill(
            mask == 0,      # replace 0's
            float("-inf")   # with -inf
        )

        # 4. Apply softmax over the key-token dimension
        scores = torch.softmax(attention_scores, dim=2)
        print(scores)

        # 5. Return (scores @ V) rounded to 4 decimal places
        return torch.round((scores @ V), decimals=4)

