import numpy as np
import pandas as pd
from collections import defaultdict

class UnigramLM:
    def __init__(self, tokenizer):
        """Simple unigram model that supports token-level and word-level calculations.

        Args:
            tokenizer (AutoTokenizer): Should be the tokenizer of the model used in surprisal calculations.
        """
        self.model = defaultdict(float)
        self.tokenizer = tokenizer
        self.uid_unit = None
        self.dataset = None
        self.tokens = None
        
    def fit(self, text, uid_unit="token"):
        """Fit the Unigram model on given text.

        Args:
            text (str): text to fit
            uid_unit (str, optional): Unit to fit, either 'token' or 'word'. Defaults to "token".

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        if uid_unit == "token":
            self._fit_tokens(tokens)
        elif uid_unit == "word":
            self._fit_words(tokens)
        else:
            raise ValueError(f"UID Unit {uid_unit} not supported.")
        self.dataset = text
        self.tokens = tokens
        self.uid_unit = uid_unit
        return self
        
    def _fit_tokens(self, tokens):
        total_tokens = len(tokens)
        for tok in tokens:
            self.model[tok] += 1 / total_tokens
        return self.model
    
    def _fit_words(self, tokens):
        words = self._convert_tokens_to_words(tokens)
        total_words = len(words)
        for word in words:
            self.model[word] += 1 / total_words
        return self.model
    
    def _convert_tokens_to_words(self, tokens):
        words = []
        curr_word = ""
        for tok in tokens:
            if tok in self.tokenizer.all_special_tokens:
                continue
            elif tok.startswith("Ġ") or tok.startswith("_"):
                words.append(curr_word)
                curr_word = tok.strip("Ġ").strip("_")
            else:
                curr_word += tok
        if curr_word:
            words.append(curr_word)
        return words
    
    def __call__(self, text):
        """Get the unigram probabilities of each unit and the total probability of the given text.

        Args:
            text (str, List[str]): string of full text or list of tokens to process

        Raises:
            ValueError: If model has invalid unit, or is called before fitting

        Returns:
            (List[float], float): probabilities of each unit and the total probability (product of each unit's unigram probability)
        """
        if len(self.model) < 1:
            raise ValueError("Tried to call before fitting unigram model")
        
        if isinstance(text, str):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        elif isinstance(text, list):
            tokens = text
            
        if self.uid_unit == "token":
            result = np.array([self.model.get(tok, 0) for tok in tokens])
        elif self.uid_unit == "word":
            words = self._convert_tokens_to_words(tokens)
            result = np.array([self.model.get(word, 0) for word in words])
        else:
            raise ValueError(f"UID Unit {self.uid_unit} not supported.")
        
        return result, result.prod()