import conllu
import deplacy
from pathlib import Path
from pyinflect import getAllInflections, getInflection
from conllu import TokenList
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.units import *

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def load_lm(model_name="distilgpt2", device=None):
    """ Load the requested language model and corresponding tokenixer from the transformers library.

    Args:
        model_name (str, optional): Name of LM to load. Defaults to "distilgpt2".
        device (str, optional): Device on which to save the model (ex: 'cuda').

    Returns:
        Tuple(tokenizer, model, device): tuple of loaded tokenizer and model, 
            along with the device they're saved on.
    """
    # note - using distilgpt for fast prototyping, use gpt-2 for final
    assert device is not None, "Please specify device."
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    model.eval()

    model.to(device)
    return tokenizer, model, device

def render_tree(sent):
    """Render a dependency tree with deplacy.

    Args:
        sent (TokenList | Sentence | list): sentence to render
    """

    if isinstance(sent, Sentence):
        sent = TokenList([dict(w) for w in sent])
    elif isinstance(sent, list):
        sent = TokenList([dict(w) for w in sent])
    return deplacy.render(sent.serialize())

def get_batches(items, batch_size, device="cpu"):
    """Batch a list of items according to a given batch size.

    Args:
        items (_type_): _description_
        batch_size (_type_): _description_
    """
    num_batches = int(np.ceil(len(items) / batch_size))
    batched = []
    for i in range(num_batches):
        start_idx = i * batch_size
        batch = items[start_idx:start_idx + batch_size]
        batched.append(batch)
    return batched


def plot_token_surprisal(tokens,
                         surprisals,
                         title=None,
                         figsize=(12, 4),
                         rotate_tokens=45,
                         show=True,
                         save_path=None):
    """Plot token-by-token surprisal values.

    Args:
        tokens (List[str]): token sequence
        surprisals (List[float]): surprisal value per token
        title (str, optional): plot title
        figsize (Tuple[int, int], optional): matplotlib fig size
        rotate_tokens (int, optional): x tick label rotation
        show (bool, optional): call plt.show()
        save_path (str, optional): path to save figure
    """
    if len(tokens) != len(surprisals):
        raise ValueError(
            f"Token/surprisal length mismatch: {len(tokens)} vs {len(surprisals)}"
        )

    x = np.arange(len(tokens))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, surprisals, marker="o", linewidth=1.5)

    ax.set_xlabel("Token")
    ax.set_ylabel("Surprisal (bits)")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=rotate_tokens, ha="right")
    ax.grid(axis="y", alpha=0.3)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def tokens_to_words(tokens, tokenizer):
    result = []
    curr_word = ""
    for tok in tokens:
        if tok in tokenizer.all_special_tokens:
            continue
        elif tok.startswith("Ġ") or tok.startswith("_"):
            if curr_word:
                result.append(curr_word)
            curr_word = tok
        else:
            curr_word += tok
    if curr_word:
        result.append(curr_word)
    return result

def is_animate(word):
    """Determines if the given word is animate via. a majority vote of the 
        word's possible definitions. If a tie occurs, it is up to chance.

    Args:
        word (str): Word in question.

    Returns:
        bool: True if animate, False otherwise.
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    animate_defs = 0
    inanimate_defs = 0
    for syn in synsets:
        animate = any(anim in syn.lexname() for anim in ['person', 'animal'])
        if animate:
            animate_defs += 1
        else:
            inanimate_defs += 1
    if animate_defs == inanimate_defs:
        return random.random() >= 0.5
    return animate_defs > inanimate_defs

def is_definite(const_word):
    """Determines if the given word is definite given its context. 
        Requires that the word have children from the dependency tree.

    Args:
        const_word (Word): word object containing children of the word in 
            dependency tree.

    Returns:
        bool: True if the word is definite, determined by presence of a 
            definite article ("The") or possessive marker.
    """
    for word in const_word['children']:
        if word is None:
            continue
        if word['feats'] is not None and word.get('feats', 'NA').get('Definite', 'NA') == 'Def':
            return True
        if word['deprel'] == 'nmod:poss':
            return True
    return False