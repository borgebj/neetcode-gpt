import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        result = []

        for i in range(new_chars):
            generator.set_state(initial_state)

            # limit context to 'conext_length'
            context = context[:, -context_length:]

            # forward pass
            logits = model(context)

            # last positions logits 
            # :     keep every batch row
            # -1    select last sequence position
            # :     keep all vocabulary scores
            last_digit = logits[:, -1, :]

            # predict (probabilities)
            probabilities = nn.functional.softmax(last_digit, dim=-1)

            # get next token based on probabilities
            next_token = torch.multinomial(probabilities, 1, generator=generator)
            next_char = int_to_char[next_token.item()]

            # append token to context
            context = torch.cat((context, next_token), dim=-1)
            result.append(next_char)

        return ''.join(result)
