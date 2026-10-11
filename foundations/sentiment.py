import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: 
        #   Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid

        # Embedding layer
        # - manages embedding matrix
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        
        # Linear layer
        # - z = XW + b
        self.linear_layer = nn.Linear(16, 1)

        # Activation layer
        # - a = sigmoid(z)
        self.sigmoid_layer = nn.Sigmoid()


    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # x: token-ID tensor of shape (batch_size, sequence_length)

        # B = batch size: no. sentences processed at once
        # T = seq. length: no. tokens in each sentence (incl. padding)
        # 16 = embed dimesnsion: no. values representing each token

        # Convert token IDs (x) into embedding vectors: (B, T, 16)
        embeddings = self.embedding_layer(x)

        # Average token embeddings into one vector per sentence: (B, 16)
        #   'bag-of-words'
        sentence_vector = embeddings.mean(dim=1)

        # Pre-activation: z = Wx + b -> (B, 1)
        z = self.linear_layer(sentence_vector)
        
        # Activation: a = sigmoid(z) -> (B, 1)
        a = self.sigmoid_layer(z)

        return torch.round(a, decimals=4)
