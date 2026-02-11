import conllu
import deplacy
from pyinflect import getAllInflections, getInflection
from conllu import TokenList
from units.sentence import Sentence
import numpy as np
import torch

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