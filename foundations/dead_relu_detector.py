import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []

        # disables gradients
        with torch.no_grad():
            for layer in model.children():
                
                # layer forward
                x = layer(x)

                # Relu: computes fraction of neurosn whose out is 0 for all
                if isinstance(layer, nn.ReLU):
                    print(layer)

                    # true/false for each neuron
                    # A neuron is dead if it outputs 0 for ALL samples in the batch.
                    dead = (x == 0).all(dim=0)

                    # fraction of all dead neurons
                    dead_neurons = dead.float().mean()
                    
                    dead_fractions.append(dead_neurons)

        return dead_fractions


    def suggest_fix(self, dead_fractions: List[float]) -> str:
        highest_fraction = float("-inf")

        for i, layer_fraction in enumerate(dead_fractions):
            highest_fraction = max(layer_fraction, highest_fraction)

            # severe: half the layer is useless, switch activation
            if layer_fraction > 0.5:
                return 'use_leaky_relu'

            # early-layer death propagates, re-init weights
            if i == 0 and layer_fraction > 0.3:
                return 'reinitialize'

        # check if strictly increasing [a > b > c > d ...]
        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]   # compare neighbors
            for i in range(len(dead_fractions) - 1)     # for all layers
        )

        # learning rate is pushing deeper layers into death
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'

        # minor dead neurons are normal and OK
        if highest_fraction < 0.1:
            return 'healthy'
        
        return 'healthy'
            