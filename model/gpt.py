import torch
import torch.nn as nn
from torchtyping import TensorType

class GPT(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        
        # converts token IDs int vectors 
        self.embedding = nn.Embedding(vocab_size, model_dim)

        # gives each position in thesequence its own vector
        self.position = nn.Embedding(context_length, model_dim)

        # runs each transformer block sequentially
        self.transformer_blocks = nn.Sequential()

        for i in range(num_blocks):
            self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))

        # Normalizes the final token representation before prediction
        self.layernorm = nn.LayerNorm(model_dim)

        # Converts each token representation into one score per vocabulary token
        self.vocab_projection = nn.Linear(model_dim, vocab_size)


    def forward(self, context: TensorType[int]) -> TensorType[float]:
        torch.manual_seed(0)

        # Embed tokens and add positional information
        # context:  (batch_size, sequence length)
        # x:        (batch_size, sequence_length, model_dim)
        x = self.embedding(context)                     # token [0] -> embedding [-0.9, 0.5, 0.2, -0.1]
        positions = torch.arange(context.shape[1])      # positions = [0, 1, 2, 3, ...]
        x = x + self.position(positions)

        # pass through every transformer block in sequence
        x = self.transformer_blocks(x)

        # normalize then produce logits for every position
        x = self.layernorm(x)
        x = self.vocab_projection(x)

        # returns raw scores, not probabilities
        return torch.round(x, decimals=4)


    # Do NOT modify the code below this line
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    torch.manual_seed(0)

                    # learns separate projections for keys, queries, values
                    self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                    self.value_gen = nn.Linear(model_dim, head_size, bias=False)
                
                def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    k = self.key_gen(embedded)
                    q = self.query_gen(embedded)
                    v = self.value_gen(embedded)

                    # compare queries with keys to calculate attention scores
                    scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    # mask future positions so tokens cannot attend to the future
                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = lower_triangular == 0
                    scores = scores.masked_fill(mask, float('-inf'))

                    # convert scores into weights, then combine the value sectors
                    scores = nn.functional.softmax(scores, dim = 2)
                    return scores @ v
                
                
            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                torch.manual_seed(0)

                # create independent attention heads
                self.att_heads = nn.ModuleList()

                for i in range(num_heads):
                    self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

                # combine the concatenated head outputs into model_fim features
                self.output_proj = nn.Linear(model_dim, model_dim, bias=False)


            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                head_outputs = []

                # each head independently learns relationships between tokens
                for head in self.att_heads:
                    head_outputs.append(head(embedded))

                # joins the heads along the feature dimension
                concatenated = torch.cat(head_outputs, dim = 2)
                return self.output_proj(concatenated)
        

        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                torch.manual_seed(0)

                # expand the feature dimension, apply a nonlinearity, then reduce it
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2
            
            def forward(self, x: TensorType[float]) -> TensorType[float]:
                torch.manual_seed(0)

                # processes each token representation independently
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)

            self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.linear_network = self.VanillaNeuralNetwork(model_dim)

            # normalize inputs to each sublayer
            self.first_norm = nn.LayerNorm(model_dim)
            self.second_norm = nn.LayerNorm(model_dim)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)

            # residual connection: preserve the input while adding attention output
            embedded = embedded + self.attention(self.first_norm(embedded)) # skip connection

            # apply the feed-forward network and its output through another skip
            embedded = embedded + self.linear_network(self.second_norm(embedded)) # another skip connection

            return embedded
