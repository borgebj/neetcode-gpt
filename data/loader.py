import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        torch.manual_seed(0)

        # 'batch_size' random starting indices for each batch
        start = torch.randint(len(data) - context_length, (batch_size,))

        data = torch.tensor(data)

        offsets = torch.arange(context_length)

        # input sequence
        X = data[start[:, None] + offsets]
        # input sequence, shifted +1 
        Y = data[start[:, None] + offsets + 1]

        return X, Y