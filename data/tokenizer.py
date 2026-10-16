from typing import List


class Solution:

    def get_pair_frequencies(self, corpus) -> Counter:
        # counts freqencies
        pairs = Counter()
        for i in range(len(corpus) - 1):
            pair = (corpus[i], corpus[i + 1])
            pairs[pair] += 1
        
        return pairs

    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # corpus:     input text, e.g. "abcabc"
        # num_merges: how many merges, e.g. 2, "abcabc" -> "[abc","abc"]
        
        tokens = list(corpus)
        merges = []

        print("corpus: ", corpus)
        print()

        for _ in range(num_merges):

            # 1. count pairs in current tokens
            pairs = self.get_pair_frequencies(tokens)

        
            # 2. finds most freq pair, breaking ties lexicographically
            highest = max(pairs.values())
            highest_pairs = [
                pair for pair,count in pairs.items() 
                if count == highest
            ]
            most_freq = min(highest_pairs)

            # 3. Merge all non-overlapping occurences left to right
            new_tokens = []

            i = 0
            while i < len(tokens):
                # checks if token and next forms highest pair
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == most_freq:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # 4. record the merge and update tokens for next iteration
            merges.append(list(most_freq))
            tokens = new_tokens

            print()
            print("tokens: ", tokens)
            print("highest:", most_freq)
            print("new:    ", new_tokens)

        print()
        print(tokens)
        
        return merges
