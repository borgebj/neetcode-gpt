import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

        combined = (positive + negative)

        # Builds vocabulary: collect all unique words, sort them, assign integer IDs starting at 1

        # all tokens in the vocab:   (dog, cat, ...)
        vocab = sorted({word for sentence in combined for word in sentence.split()})

        # tokesn to id:  {"cat":1, "dog":2, ...}
        token_to_id = {token:(i+1) for i, token in enumerate(vocab)}

        # words to ids:  [dog is a cat] -> [3 4 1 2]
        encoded  = [
            torch.tensor([token_to_id[token] for token in lst.split()])
            for lst in combined
        ]

        # Pads shorter sequences with 0s to match the longest: [1, 2], [3] -> [[1, 2], [3, 0]]
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)

