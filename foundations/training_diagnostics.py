import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # info
        # x = (8, 32) = 8 samples, 32 features

        stats = []

        # 'no gradient' calculations
        with torch.no_grad():
            
            # forward pass
            # y_hat = model.forward(x)

            for layer in model.children():

                x = layer(x)

                # for pre-activations (z)
                if isinstance(layer, nn.Linear):
                    mean = x.mean()
                    std = x.std()

                    # Did each neuron fire (> 0) for at least one sample?
                    fires = (x > 0).any(dim=0)

                    # Fraction of neurons that never fired
                    dead_fraction = (~fires).float().mean()

                    stats.append({
                        "mean": round(mean.item(), 4),
                        "std": round(std.item(), 4),
                        "dead_fraction": round(dead_fraction.item(), 4)
                    })

        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:

        # clears gradients stored
        model.zero_grad()

        # forward
        predictions = model(x)

        # MSE loss
        loss = nn.MSELoss()(predictions, y)

        # backward
        loss.backward()

        stats = []

        for layer in model.children():
            if isinstance(layer, nn.Linear):

                # weight gradients
                grad = layer.weight.grad

                # gradient stats
                mean = grad.mean()
                std = grad.std()
                norm = torch.norm(grad)

                stats.append({
                    "mean": round(mean.item(), 4),
                    "std": round(std.item(), 4),
                    "norm": round(norm.item(), 4)
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        
        # 1. Dead neurons
        for layer_stats in activation_stats:
            if layer_stats["dead_fraction"] > 0.5:
                return 'dead_neurons'

        # 2. Exploding gradients
        for layer_stats in gradient_stats:
            if layer_stats["norm"] > 1000:
                return 'exploding_gradients'

        # 3. Vanishing gradients
        if gradient_stats[-1]["norm"] < 1e-5:
            return 'vanishing_gradients'

        # 4. Activation std
        for layer_stats in activation_stats:
            if layer_stats["std"] < 0.1:
                return 'vanishing_gradients'

            if layer_stats["std"] > 10.0:
                return 'exploding_gradients'

        # 5. healthy
        return 'healthy'
