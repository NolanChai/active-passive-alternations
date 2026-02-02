import conllu
from pyinflect import getAllInflections, getInflection
from .units.word import Word
# from units.sentence import Sentence

def switch_pronoun(const):
    """ Switch relevant pronouns from subject to object.

    Args:
        const (list): list of Word objects to process
    """
    pron_key = {
        "i": "me",
        "he": "him",
        "she": "her",
        "we": "us",
        "they": "them",
        "me": "I",
        "him": "he",
        "her": "she",
        "us": "we",
        "them": "they"
    }
    for w in const:
        # print(w['form'], w['deprel'])
        if w['deprel'] in ['obl:agent', 'nsubj:pass'] and w['form'].lower() in pron_key:
            w['form'] = pron_key[w['form'].lower()]