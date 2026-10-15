import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # embedding_dim:  n-dimensions / features each embedding have
        # attention_dim:  output feature size from attention layer
        
        # creates 'num_heads' SingleHeadAttention instances
        # Each head learns its own K,Q,V projection, allowing learning of different relationships between toknes
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(
                self.SingleHeadAttention(
                    embedding_dim, 
                    attention_dim // num_heads)
                )

        # output projection
        # mixes info from heads after outputs has been concatenated
        self.out_projection = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded:  embeddings for all tokens in input sequence
        
        # 1. runs every head independently
        head_outputs = [head(embedded) for head in self.heads]

        # 2. Concatenates along feature dimension
        combined_outs = torch.cat(head_outputs, dim=-1) # last, e.g. 4 in (2,2,4)

        # 3. mixes the head outputs (applies linearity)
        output = self.out_projection(combined_outs)
        
        return torch.round(output, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
