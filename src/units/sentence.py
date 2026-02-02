import pyinflect
from pyinflect import getAllInflections, getInflection
import conllu
from conllu import Token, TokenList
from .word import Word

def build_subtree(root,
                  upos: list = None,
                  deprel: list = None):
    """return all words under the root in the dependency tree, subject to filters.

    Args:
        root (Word): The root word of the subtree to build.

    Raises:
        ValueError: If root is not a Word object.

    Returns:
        list[Word]: The list of Word objects in the subtree, ordered by ID.
    """
    if not isinstance(root, Word):
        raise ValueError("root must be a Word object")

    subtree = []
    to_visit = [root]

    while to_visit:
        node = to_visit.pop()
        subtree.append(node)
        if upos or deprel:
            to_visit += [child for child in node['children']
                        if child['upos'] in upos
                        or child['deprel'] in deprel]
        else:
            to_visit += node['children']
    subtree = sorted(subtree, key=lambda w: w['id'])
    return subtree

class Sentence(list[Word]):
    """
    Wrapper class for a list of Word objects to represent a sentence.
    """
    def __init__(self, tokens: list[dict]) -> None:
        super().__init__([Word(token_dict=t) for t in tokens
                          if t['lemma'] != '_'])
        self.root = next((word for word in self if word['head'] == 0), None)
        if isinstance(tokens, TokenList):
            self.text = tokens.metadata['text']
        elif isinstance(tokens, Sentence):
            self.text = tokens.text
        else:
            self.text = ' '.join([word['form'] for word in self])
        self.is_passive = (any(word['deprel'] == 'obl:agent' for word in self) 
                           and any(word['deprel'] == 'nsubj:pass' for word in self))
        
        self.populate_children()
        
    def populate_children(self):
        """Populate the children attribute for each Word in the sentence."""
        id_to_word = {word['id']: word for word in self}
        for word in self:
            head_id = word['head']
            if head_id:
                head_word = id_to_word[head_id]
                head_word['children'].append(word)
                
class PassiveSentence(Sentence):
    def __init__(self, tokens: list[dict]) -> None:
        super().__init__(tokens)
        if not self.is_passive:
            raise ValueError("The provided sentence is not passive.")
        
        # Find passive subject
        self.passive_subject, self.passive_subject_word = self.find_pass_subj()
        
        # Find main verb
        self.verb, self.verb_word = self.find_verb()
        
        # Find agent
        self.agent, self.agent_word = self.find_pass_agent()
        
    def find_pass_subj(self):
        """
        Extract the passive subject + constituent from this sentence.
        
        Returns:
            tuple: (list of Word objects in the passive subject subtree, Word object of the passive subject)
        """

        # Passive subject takes on the 'nsubj:pass' deprel tag
        # TODO: Support multiple passive subjects?
        subj = next(filter(lambda w: w['deprel'] == "nsubj:pass", self), None)
        if subj is None:
            return None, None

        subj_subtree = build_subtree(subj)
        return subj_subtree, subj

    def find_verb(self):
        """
        Extract the verb + constituent from this sentence.
        
        Returns:
            tuple: (list of Word objects in the verb subtree, Word object of the verb)
        """
        # Get the verb that is the head of the passive subject
        verb = None
        if self.passive_subject_word['head'] > 0:
            verb = next(filter(lambda w: w['id'] == self.passive_subject_word['head'], self), None)
        if verb is None or verb['upos'] != "VERB":
            return None, None
        verb_subtree = build_subtree(verb,
                                upos=['ADV'],
                                deprel=['aux', 'aux:pass']) # pass aux will be removed later
        
        # remove adverbs that come after the verb
        verb_subtree = list(filter(lambda w: w['upos'] != 'ADV' or w['id'] < verb['id'], verb_subtree))
        return verb_subtree, verb
    
    def find_pass_agent(self):
        """
        Extracts the passive agent from this sentence.

        Returns:
            tuple: (list of Word objects in the passive agent subtree, Word object of the passive agent)
        """
        # Agent has the deprel tag 'obl:agent' and is attached to the same verb
        agent = next(
            filter(
                lambda w: w['deprel'] == "obl:agent"
                    and w in self.verb_word['children'],
                self
                ),
            None
        )
        # In case agent isn't found that way, look for 'obl' tag and a "by" phrase
        if agent is None:
            for w in self:
                if w['deprel'] == "obl" and w in self.verb_word['children']:
                    case_children = [c for c in w['children'] if c['deprel'] == "case"]
                    if any(c['form'].lower() == "by" for c in case_children):
                        agent = w
                        break
        if agent is None:
            return None, None

        # extract subtree
        agent_subtree = build_subtree(agent)

        # ids -> words
        return agent_subtree, agent