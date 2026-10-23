from typing import List, Dict

class Solution:
    def greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0

        # go through all digits, build longest substring
        while i < len(text):
            found = False
            
            # go backwards look for longest valid substring
            for j in range(len(text), i, -1):
                substr = text[i:j]
                
                if substr in vocab:
                    
                    found = True
                    tokens.append(substr)
                    
                    i += len(substr)
                    break
                        
            if not found:
                tokens.append(text[i])
                i += 1
            
        return tokens

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenizes each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.

        result = []

        # go through all numbers
        for num in numbers:
            text = str(num)
            tokens = self.greedy_tokenize(text, vocab)
            result.append(tokens)

        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Counts how many tokens the text uses with greedy tokenization.
        return len(self.greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Computes tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        tokens = self.greedy_tokenize(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)
