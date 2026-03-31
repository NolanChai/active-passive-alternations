import conllu
import deplacy
from pathlib import Path
from pyinflect import getAllInflections, getInflection
from conllu import TokenList
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.units import *

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