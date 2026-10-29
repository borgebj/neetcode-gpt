import torch
import torch.nn as nn
import torch.nn.functional as F


class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)
    
            start = torch.randint(len(data) - context_length, (batch_size,))

            offsets = torch.arange(context_length)
            x = data[start[:, None] + offsets]
            y = data[start[:, None] + offsets + 1]

            # runs on data (x), gets loss with targets (y)
            logits = model(x)
            B, T, C = logits.shape
            logits_flat = logits.reshape(-1, C)
            targets_flat = y.reshape(-1)

            loss = F.cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()   # clears prev. gradients
            loss.backward()         # calculate current gradients
            optimizer.step()        # update model weights

        return round(loss.item(), 4)
