import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid

        # hidden 1: 784 input -> 512 neurons
        self.layer1 = nn.Linear(784, 512)

        # hidden activation
        self.relu = nn.ReLU()   

        self.dropout = nn.Dropout(0.2)

        # output:  512 neurons -> 10 outputs (digits)
        self.layer2 = nn.Linear(512, 10)

        # output activation
        self.sigmoid = nn.Sigmoid()


    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)

        # layer 1 forward + activation
        x = self.layer1(images)
        x = self.relu(x)

        # dropout randomly sets a fraction of activations to zero during training each step
        x = self.dropout(x)

        # output forward + activation
        x = self.layer2(x)
        x = self.sigmoid(x)

        return x.round(decimals=4)
