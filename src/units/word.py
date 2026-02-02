import pyinflect
from pyinflect import getAllInflections, getInflection
import conllu
from conllu import Token

class Word(Token):
    """
    Wrapper class for conllu.Token to represent a word in a sentence.
    Includes inflection of word and children of word in the dependency tree.
    """
    def __init__(self,
                 token_dict: dict,
                 inflection: str = None,
                 children: list = None) -> None:
        super().__init__(token_dict)
        if inflection:
            self['inflection'] = inflection
        elif self['xpos'] != '_':
            self['inflection'] = self['xpos']
        else:
            self['inflection'] = self.read_inflection(pos_type=self['upos'][0])
        self['children'] = children if children else []
        
    def read_inflection(self, 
                        pos_type='V'):
        """Get the inflection of a word.

        Args:
            word (Word): word to get inflection for
            pos_type (str, optional): part-of-speech type: 'V' for verb, 'N' for noun, etc. Defaults to 'V'.

        Returns:
            str: the inflection tag of the word, or None if not found
        """
        inflections = getAllInflections(self['lemma'], pos_type)
        for tag, forms in inflections.items():
            if self['form'] in forms:
                return tag
        # naively assume plural if ends in 's' for proper nouns
        if self['upos'] == 'PROPN':
            return 'NNS' if self['form'][-1] == 's' else 'NN'
        return None

    def reinflect(self, new_inflection: str):
        """Reinflect the word to a new inflection.

        Args:
            new_inflection (str): the target inflection tag
        """
        new_form = getInflection(self['lemma'], new_inflection)
        if new_form:
            self['form'] = new_form[0]
            self['inflection'] = new_inflection
            
    def deep_copy(self):
        """Create a deep copy of the Word object. Does not include children.

        Returns:
            Word: A new Word object that is a deep copy of the original.
        """
        return Word(token_dict=dict(self), 
                    inflection=self['inflection'])