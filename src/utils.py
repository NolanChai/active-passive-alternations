import conllu
import deplacy
from pyinflect import getAllInflections, getInflection
from conllu import TokenList
from .units.sentence import Sentence

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
