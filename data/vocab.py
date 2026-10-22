from math import e
from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        vocab = sorted(set(text))

        # string-to-int
        stoi = {ch: i for i, ch in enumerate(vocab)}

        # int-to-string
        itos = {i: ch for ch, i in stoi.items()}

        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # converts string to ids:  "hello" > [1,0,2,2,3]
        return [stoi[ch] for ch in text]


    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # converts ids to string:  [1,0,2,2,3] > "hello"
        return  ''.join(itos[id] for id in ids)
        
