import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        
        # splits into tokens "the cat was" -> [the, cat, was]
        tokens = raw_dataset.split(" ")
        
        start = torch.randint(len(tokens) - context_length, (batch_size,)).tolist()

        x = []
        y = []

        for i in range(batch_size):
            xi = tokens[start[i]:start[i] + context_length]
            yi = tokens[start[i] + 1:start[i] + context_length + 1]

            x.append(xi)
            y.append(yi)

        return x, y
