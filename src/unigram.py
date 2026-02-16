import numpy as np
import pandas as pd
from collections import defaultdict

class UnigramLM:
    def __init__(self, tokenizer):
        self.model = defaultdict(float)
        self.tokenizer = tokenizer
        self.uid_unit = None
        self.dataset = None
        self.tokens = None
        
    def fit(self, text, uid_unit="token"):
        self.dataset = text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        self.tokens = tokens
        
        if uid_unit == "token":
            self._fit_tokens(tokens)
        elif uid_unit == "word":
            self._fit_words(tokens)
        self.uid_unit = uid_unit
        return self
        
    def _fit_tokens(self, tokens):
        total_tokens = len(tokens)
        for tok in tokens:
            self.model[tok] += 1 / total_tokens
        return self.model
    
    def _fit_words(self, tokens):
        total_words = 0
        curr_word = ""
        for tok in tokens:
            if tok in self.tokenizer.all_special_tokens:
                continue
            elif tok.startswith("Ġ") or tok.startswith("_"):
                self.model[curr_word] += 1.
                total_words += 1
                curr_word = tok
            else:
                curr_word += tok
        self.model = {key: value / total_words 
                      for key, value in self.model.items()}
        return self.model